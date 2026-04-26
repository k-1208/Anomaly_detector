import pandas as pd

from models.metrics import ResourceMetrics
from utils.ml_model import MLAnomalyScorer

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("./enhanced_normal_resources.csv")
fleet = []

for _, row in df.iterrows():
    r = ResourceMetrics(
        resource_id=row["resource_id"],
        cpu_avg=row["cpu_avg"],
        cpu_p95=row["cpu_p95"],
        memory_avg=row["memory_avg"],
        network_pct=row["network_pct"],
        internet_facing=row.get("internet_facing", False),
        identity_attached=row.get("identity_attached", False),
    )
    fleet.append(r)

print(f"[INFO] Loaded {len(fleet)} resources")

# =========================
# 3. TRAIN MODEL
# =========================
scorer = MLAnomalyScorer()
scorer.fit(fleet)

# =========================
# 4. VALIDATE MODEL (VERY IMPORTANT)
# =========================
print("\n[INFO] Validating on training data...")

false_positives = []

for r in fleet:
    out = scorer.score(r)

    # ❗ This is what YOU care about
    if out.is_anomalous:
        false_positives.append((r.resource_id, out.final_score))

print(f"[INFO] False positives: {len(false_positives)}")

# print few errors
for fp in false_positives[:5]:
    print("[MODEL ISSUE]", fp)

# =========================
# 5. TEST WITH NEW DATA
# =========================
print("\n[INFO] Testing on new samples...")

test_cases = [
    ResourceMetrics("t-1", 5, 10, 80, 10, True, True),
    ResourceMetrics("t-2", 90, 98, 50, 60, False, False),
    ResourceMetrics("t-3", 20, 30, 30, 90, True, False),
    ResourceMetrics("t-4", 45, 60, 50, 40, False, False),  # normal
]

for r in test_cases:
    out = scorer.score(r)

    print("\n========================")
    print(f"Resource: {r.resource_id}")
    print(f"Anomalous: {out.is_anomalous}")
    print(f"Score: {out.final_score}")
    print(f"Confidence: {out.confidence}")
    print(f"Type: {out.anomaly_type}")
    print(f"Reason: {out.reason}")