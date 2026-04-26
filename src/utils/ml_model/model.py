"""
ML Anomaly Scorer — complete implementation
Three methods: Z-score + Isolation Forest + Peer comparison
Weighted fusion → anomaly type classification
"""

import json
import math

from models.metrics import ResourceMetrics
from models.scoring import ScorerOutput

# ─────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# Derive signals the raw metrics don't expose
# ─────────────────────────────────────────────

def extract_features(r: ResourceMetrics) -> dict:
    cpu_safe = r.cpu_avg if r.cpu_avg > 0 else 0.01
    mem_safe = r.memory_avg if r.memory_avg > 0 else 0.01

    return {
        "cpu_avg":        r.cpu_avg,
        "cpu_p95":        r.cpu_p95,
        "memory_avg":     r.memory_avg,
        "network_pct":    r.network_pct,
        # Derived ratios — the key signals rules miss
        "cpu_p95_delta":  r.cpu_p95 - r.cpu_avg,          # burst width
        "net_cpu_ratio":  r.network_pct / cpu_safe,        # net anomaly signal
        "mem_cpu_ratio":  r.memory_avg / cpu_safe,         # over-provisioned signal
        "utilisation":    (r.cpu_avg + r.memory_avg) / 2,  # overall load
    }

FEATURE_KEYS = [
    "cpu_avg", "cpu_p95", "memory_avg", "network_pct",
    "cpu_p95_delta", "net_cpu_ratio", "mem_cpu_ratio", "utilisation"
]

# ─────────────────────────────────────────────
# 3. METHOD A — Z-SCORE SCORER  (weight 0.30)
#
# For each metric: z = (x - mean) / std
# Score = average of clamped |z| values across all features
# Catches: single-metric extremes (very high CPU, very low util)
# ─────────────────────────────────────────────

class ZScoreScorer:
    def __init__(self):
        self.means: dict = {}
        self.stds:  dict = {}
        self.fitted = False

    def fit(self, fleet: list[dict]):
        """Learn mean and std for each feature across the fleet."""
        for key in FEATURE_KEYS:
            vals = [f[key] for f in fleet if key in f]
            n = len(vals)
            mean = sum(vals) / n
            std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
            self.means[key] = mean
            self.stds[key]  = std if std > 0 else 1.0
        self.fitted = True

    def score(self, features: dict) -> tuple[float, list[str]]:
        """Returns anomaly score 0-1 and list of flagged features."""
        if not self.fitted:
            raise RuntimeError("Call fit() before score()")

        z_scores = {}
        for key in FEATURE_KEYS:
            val = features.get(key, 0)
            z = abs(val - self.means[key]) / self.stds[key]
            z_scores[key] = z

        # Score = mean of clamped z-scores (clamp at 3 → maps to 1.0)
        raw = sum(min(z / 3.0, 1.0) for z in z_scores.values()) / len(z_scores)
        flagged = [k for k, z in z_scores.items() if z > 2.5]
        return round(min(raw, 1.0), 4), flagged


# ─────────────────────────────────────────────
# 4. METHOD B — ISOLATION FOREST  (weight 0.45)
#
# Pure-Python implementation (no sklearn dependency).
# Idea: anomalies are isolated with fewer random splits.
# Short average path length → high anomaly score.
# Catches: multi-metric combinations that are globally unusual
# ─────────────────────────────────────────────

import random

class IsolationTree:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value   = None
        self.left  = None
        self.right = None
        self.size  = 0
        self.is_leaf = False

    def fit(self, data: list[list[float]], depth: int = 0):
        self.size = len(data)
        n_features = len(FEATURE_KEYS)

        if depth >= self.max_depth or len(data) <= 1:
            self.is_leaf = True
            return self

        self.split_feature = random.randint(0, n_features - 1)
        col = [row[self.split_feature] for row in data]
        col_min, col_max = min(col), max(col)

        if col_min == col_max:
            self.is_leaf = True
            return self

        self.split_value = random.uniform(col_min, col_max)
        left_data  = [row for row in data if row[self.split_feature] < self.split_value]
        right_data = [row for row in data if row[self.split_feature] >= self.split_value]

        self.left  = IsolationTree(self.max_depth).fit(left_data,  depth + 1)
        self.right = IsolationTree(self.max_depth).fit(right_data, depth + 1)
        return self

    def path_length(self, row: list[float], depth: int = 0) -> float:
        if self.is_leaf:
            return depth + _c(self.size)
        if row[self.split_feature] < self.split_value:
            return self.left.path_length(row, depth + 1)
        return self.right.path_length(row, depth + 1)

def _c(n: int) -> float:
    """Expected path length of unsuccessful BST search (normalisation constant)."""
    if n <= 1:
        return 0.0
    return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

class IsolationForest:
    def __init__(self, n_trees: int = 50, sample_size: int = 64, max_depth: int = 8):
        self.n_trees     = n_trees
        self.sample_size = sample_size
        self.max_depth   = max_depth
        self.trees: list[IsolationTree] = []
        self.c_norm = 1.0

    def _to_vec(self, features: dict) -> list[float]:
        return [features.get(k, 0.0) for k in FEATURE_KEYS]

    def fit(self, fleet_features: list[dict]):
        vecs = [self._to_vec(f) for f in fleet_features]
        self.c_norm = _c(min(self.sample_size, len(vecs)))
        for _ in range(self.n_trees):
            sample = random.sample(vecs, min(self.sample_size, len(vecs)))
            tree = IsolationTree(self.max_depth).fit(sample)
            self.trees.append(tree)

    def score(self, features: dict) -> float:
        """Returns anomaly score 0-1. Higher = more anomalous."""
        vec = self._to_vec(features)
        avg_path = sum(t.path_length(vec) for t in self.trees) / len(self.trees)
        # s = 2^(-avg_path / c_norm): near 1 = anomaly, near 0.5 = normal
        raw = 2 ** (-avg_path / (self.c_norm if self.c_norm > 0 else 1))
        # Re-scale: 0.5 → 0.0,  1.0 → 1.0
        rescaled = max(0.0, (raw - 0.5) * 2)
        return round(min(rescaled, 1.0), 4)


# ─────────────────────────────────────────────
# 5. METHOD C — PEER COMPARISON  (weight 0.25)
#
# Mahalanobis-like distance: how far is this resource
# from the fleet centroid in standardised feature space?
# Catches: resources that look normal alone but are
# outliers within their peer group
# ─────────────────────────────────────────────

class PeerComparisonScorer:
    def __init__(self):
        self.means: dict = {}
        self.stds:  dict = {}
        self.fitted = False

    def fit(self, fleet: list[dict]):
        for key in FEATURE_KEYS:
            vals = [f[key] for f in fleet]
            n = len(vals)
            mean = sum(vals) / n
            std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
            self.means[key] = mean
            self.stds[key]  = std if std > 0 else 1.0
        self.fitted = True

    def score(self, features: dict) -> float:
        """Euclidean distance in standardised feature space, clamped to 0-1."""
        diffs = []
        for key in FEATURE_KEYS:
            z = (features.get(key, 0) - self.means[key]) / self.stds[key]
            diffs.append(z ** 2)
        dist = math.sqrt(sum(diffs))
        # Typical max distance in 8-dim space with z up to 3 ≈ 8.5
        return round(min(dist / 8.5, 1.0), 4)


# ─────────────────────────────────────────────
# 6. ANOMALY TYPE CLASSIFIER
#
# Given the final score + raw features → assign anomaly_type.
# Uses the derived ratios + score to pick the most likely type.
# ─────────────────────────────────────────────

def classify_anomaly_type(features: dict, final_score: float, resource: ResourceMetrics) -> tuple[str, list[str]]:
    signals = []
    anomaly_type = "none"

    if final_score < 0.35:
        return "none", []

    cpu     = features["cpu_avg"]
    cpu_p95 = features["cpu_p95"]
    mem     = features["memory_avg"]
    net     = features["network_pct"]
    ncr     = features["net_cpu_ratio"]   # network / cpu
    mcr     = features["mem_cpu_ratio"]   # memory / cpu
    burst   = features["cpu_p95_delta"]   # p95 - avg
    util    = features["utilisation"]

    candidates = []

    # Network anomaly: high network relative to CPU (ratio > 5 and net > 40)
    if ncr > 5 and net > 40 and cpu < 30:
        candidates.append(("network_anomaly", ncr * 0.1))
        signals.append(f"net/cpu ratio={ncr:.1f} (net={net}%, cpu={cpu}%)")

    # Metric correlation break: CPU and network diverge from expected pattern
    if abs(cpu - net) > 50 and final_score > 0.5:
        candidates.append(("metric_correlation_break", 0.7))
        signals.append(f"cpu={cpu}% net={net}% — unusual divergence")

    # Gradual drift: moderate score, no single metric in extreme zone
    if 0.35 < final_score < 0.65 and cpu < 80 and mem < 80 and net < 80:
        candidates.append(("gradual_drift", 0.5))
        signals.append(f"moderate multi-metric anomaly score={final_score:.2f} — possible slow drift")

    # Peer deviation: high score driven by peer comparison
    if final_score > 0.6 and util < 40:
        candidates.append(("peer_deviation", 0.65))
        signals.append(f"low utilisation={util:.1f}% but high anomaly score vs fleet")

    # Memory pressure: high memory / cpu ratio
    if mcr > 8 and mem > 60:
        candidates.append(("memory_pressure", mcr * 0.05))
        signals.append(f"mem/cpu ratio={mcr:.1f} (mem={mem}%, cpu={cpu}%)")

    # CPU p95 spike: large burst delta
    if burst > 40 and cpu_p95 > 80:
        candidates.append(("cpu_p95_spike", burst * 0.015))
        signals.append(f"cpu burst delta={burst:.0f}pp (avg={cpu}%, p95={cpu_p95}%)")

    if candidates:
        anomaly_type = max(candidates, key=lambda x: x[1])[0]

    return anomaly_type, signals


# ─────────────────────────────────────────────
# 7. MAIN SCORER — assembles all three methods
# ─────────────────────────────────────────────

WEIGHTS = {"z_score": 0.30, "isolation_forest": 0.45, "peer_comparison": 0.25}
ANOMALY_THRESHOLD = 0.40   # above this → is_anomalous = True

class MLAnomalyScorer:
    def __init__(self):
        self.z_scorer    = ZScoreScorer()
        self.if_scorer   = IsolationForest(n_trees=80, sample_size=64)
        self.peer_scorer = PeerComparisonScorer()
        self.fitted      = False

    def fit(self, fleet: list[ResourceMetrics]):
        """Train all three sub-scorers on the fleet baseline."""
        features = [extract_features(r) for r in fleet]
        self.z_scorer.fit(features)
        self.if_scorer.fit(features)
        self.peer_scorer.fit(features)
        self.fitted = True
        print(f"[Scorer] Fitted on {len(fleet)} resources.")

    def score(self, resource: ResourceMetrics) -> ScorerOutput:
        if not self.fitted:
            raise RuntimeError("Call fit() with fleet data before scoring.")

        features = extract_features(resource)

        z_score,   flagged_features = self.z_scorer.score(features)
        if_score                    = self.if_scorer.score(features)
        peer_score                  = self.peer_scorer.score(features)

        final_score = (
            WEIGHTS["z_score"]          * z_score  +
            WEIGHTS["isolation_forest"] * if_score +
            WEIGHTS["peer_comparison"]  * peer_score
        )
        final_score = round(final_score, 4)

        is_anomalous   = final_score >= ANOMALY_THRESHOLD
        anomaly_type, signals = classify_anomaly_type(features, final_score, resource)

        # Confidence: how consistently do the three methods agree?
        scores_list = [z_score, if_score, peer_score]
        mean_s      = sum(scores_list) / 3
        variance    = sum((s - mean_s) ** 2 for s in scores_list) / 3
        agreement   = 1.0 - min(math.sqrt(variance), 0.5) * 2  # high agreement = high confidence
        confidence  = round(final_score * agreement, 4)

        if not signals:
            if is_anomalous:
                signals.append(f"aggregate score {final_score:.2f} exceeds threshold")
            else:
                signals.append("all metrics within normal range")

        reason = _build_reason(resource, features, final_score, anomaly_type, signals, flagged_features)

        return ScorerOutput(
            resource_id      = resource.resource_id,
            final_score      = final_score,
            anomaly_type     = anomaly_type,
            is_anomalous     = is_anomalous,
            confidence       = confidence,
            method_scores    = {"z_score": z_score, "isolation_forest": if_score, "peer_comparison": peer_score},
            triggered_signals= signals,
            reason           = reason,
        )


def _build_reason(r, features, score, atype, signals, flagged):
    parts = [f"Final anomaly score: {score:.2f}."]
    if signals:
        parts.append("Key signals: " + "; ".join(signals) + ".")
    if flagged:
        parts.append(f"Z-score flagged metrics: {', '.join(flagged)}.")
    if atype != "none":
        parts.append(f"Classified as '{atype}'.")
    else:
        parts.append("No specific anomaly pattern matched — resource appears normal.")
    return " ".join(parts)


# ─────────────────────────────────────────────
# 8. DEMO — run on the assignment's sample data
#    + extended test cases
# ─────────────────────────────────────────────


def _suggest(atype: str) -> str:
    return {
        "network_anomaly":         "Inspect VPC flow logs; check for unexpected egress destinations",
        "metric_correlation_break":"Investigate process list; check for rogue services or misconfigs",
        "gradual_drift":           "Set up trend alerting; check for slow memory leak or log accumulation",
        "peer_deviation":          "Compare instance type and workload against peer group; consider right-sizing",
        "memory_pressure":         "Profile heap usage; consider upgrading instance memory tier",
        "cpu_p95_spike":           "Profile burst workload; consider auto-scaling or queue-based throttling",
        "none":                    "No action required; continue monitoring",
    }.get(atype, "Review resource metrics and compare against baseline")

if __name__ == "__main__":
    random.seed(42)

    # Fleet of "normal" resources used to fit the scorers
    # In production this would be 30-90 days of historical baselines
    fleet_baseline = [
        ResourceMetrics("b-1",  45, 62, 52, 38, False, False),
        ResourceMetrics("b-2",  38, 55, 48, 42, False, True),
        ResourceMetrics("b-3",  52, 70, 60, 35, False, False),
        ResourceMetrics("b-4",  41, 58, 55, 40, False, False),
        ResourceMetrics("b-5",  48, 65, 50, 44, False, True),
        ResourceMetrics("b-6",  35, 52, 45, 36, True,  False),
        ResourceMetrics("b-7",  50, 68, 58, 41, False, False),
        ResourceMetrics("b-8",  43, 60, 53, 39, False, False),
        ResourceMetrics("b-9",  46, 63, 57, 43, False, True),
        ResourceMetrics("b-10", 39, 56, 49, 37, False, False),
    ]

    # Resources to evaluate (includes assignment samples)
    test_resources = [
        # From the assignment
        ResourceMetrics("i-1",  2,  5,  70, 10, True,  True),   # over-provisioned + security
        ResourceMetrics("i-2",  85, 98, 40, 60, False, False),   # compute thrash
        # Additional ML-detectable cases
        ResourceMetrics("i-3",  8,  12,  9,  85, True,  False),  # network anomaly
        ResourceMetrics("i-4",  5,   9, 10,   4, False, False),  # idle resource
        ResourceMetrics("i-5",  44, 46, 88,  41, False, True),   # memory pressure
        ResourceMetrics("i-6",  10, 78, 35,  38, False, False),  # cpu p95 spike (burst)
        ResourceMetrics("i-7",  42, 58, 51,  39, False, False),  # normal
        ResourceMetrics("i-8",  3,   6, 15,  92, True,  True),   # network anomaly + security
    ]

    scorer = MLAnomalyScorer()
    scorer.fit(fleet_baseline)

    print("\n" + "="*70)
    print("ML ANOMALY SCORER — RESULTS")
    print("="*70)

    results = []
    for resource in test_resources:
        out = scorer.score(resource)
        results.append(out)

        status = "ANOMALOUS" if out.is_anomalous else "NORMAL   "
        print(f"\n[{status}] {out.resource_id}")
        print(f"  Score:       {out.final_score:.3f}  (threshold: {ANOMALY_THRESHOLD})")
        print(f"  Confidence:  {out.confidence:.3f}")
        print(f"  Type:        {out.anomaly_type}")
        print(f"  Methods:     z={out.method_scores['z_score']:.3f}  "
              f"IF={out.method_scores['isolation_forest']:.3f}  "
              f"peer={out.method_scores['peer_comparison']:.3f}")
        print(f"  Reason:      {out.reason}")

    # JSON output matching the assignment spec
    print("\n" + "="*70)
    print("JSON OUTPUT (assignment format)")
    print("="*70)
    for out in results:
        payload = {
            "resource_id":    out.resource_id,
            "is_anomalous":   out.is_anomalous,
            "anomaly_type":   out.anomaly_type,
            "reason":         out.reason,
            "suggested_action": _suggest(out.anomaly_type),
            "confidence":     out.confidence,
            "ml_method_scores": out.method_scores,
        }
        print(json.dumps(payload, indent=2))