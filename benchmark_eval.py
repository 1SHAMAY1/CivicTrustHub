"""
Benchmark evaluation runner for CivicTrustHub rumor detection engine.
Executes claims from eval_set.csv directly through main.verify_claim_against_web
(or an active deployed HTTP API endpoint).

Computes:
1. Overall Accuracy (%)
2. Precision & Recall per category (TRUE / FALSE / UNVERIFIABLE)
3. False-VERIFIED Rate (Safety Metric: % of FALSE claims wrongly marked VERIFIED)
4. Abstention Rate (% of claims abstained as UNVERIFIED)
5. Full Confusion Matrix (Rows: Ground Truth, Cols: Model Predictions)
"""
import csv
import collections
import time
import main

EVAL_CSV = "eval_set.csv"

# Map 3-state engine output to ground-truth labels
VERDICT_MAP = {
    "VERIFIED": "TRUE",
    "CONTRADICTED": "FALSE",
    "UNVERIFIED": "UNVERIFIABLE"
}

def run_evaluation(api_url: str = None):
    print("=" * 70)
    print("🔬 CivicTrustHub Benchmark Evaluation Suite")
    print(f"Dataset: {EVAL_CSV}")
    print(f"Target: {'HTTP API -> ' + api_url if api_url else 'Direct Native Engine (main.py)'}")
    print("=" * 70)

    with open(EVAL_CSV, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_rows = len(rows)
    print(f"Loaded {total_rows} labeled ground-truth rumor evaluation claims.\n")

    conf_matrix = collections.Counter()
    category_stats = collections.defaultdict(lambda: {"total": 0, "correct": 0})
    errors = 0
    start_time = time.time()

    for idx, r in enumerate(rows, 1):
        claim = r["claim"]
        true_label = r["label"]
        category = r["category"]

        try:
            if api_url:
                import urllib.request, json
                req = urllib.request.Request(
                    api_url,
                    data=json.dumps({"text": claim}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            else:
                data = main.verify_claim_against_web(claim)

            raw_verdict = data.get("verdict", "UNVERIFIED")
            if raw_verdict in ("NETWORK_ERROR", "INVALID_QUERY"):
                errors += 1
                model_label = "ERROR"
            else:
                model_label = VERDICT_MAP.get(raw_verdict, "UNVERIFIABLE")

        except Exception as ex:
            print(f"[{idx}/{total_rows}] Exception on '{claim[:35]}...': {ex}")
            errors += 1
            model_label = "ERROR"

        conf_matrix[(true_label, model_label)] += 1
        category_stats[category]["total"] += 1
        if true_label == model_label:
            category_stats[category]["correct"] += 1

        print(f"[{idx:02d}/{total_rows:02d}] Truth: {true_label:<12} | Model: {model_label:<12} | Claim: \"{claim[:42]}...\"")

    duration = time.time() - start_time
    eval_total = sum(conf_matrix.values()) - errors
    correct = sum(conf_matrix[(l, l)] for l in ["TRUE", "FALSE", "UNVERIFIABLE"])

    accuracy = (correct / eval_total * 100) if eval_total > 0 else 0.0

    # 1. Critical Safety Metric: False-VERIFIED Rate
    false_claims_total = sum(conf_matrix[("FALSE", p)] for p in ["TRUE", "FALSE", "UNVERIFIABLE", "ERROR"])
    false_marked_true = conf_matrix[("FALSE", "TRUE")]
    false_verified_rate = (false_marked_true / false_claims_total * 100) if false_claims_total > 0 else 0.0

    # 2. Abstention Rate
    unverified_predictions = sum(conf_matrix[(t, "UNVERIFIABLE")] for t in ["TRUE", "FALSE", "UNVERIFIABLE"])
    abstention_rate = (unverified_predictions / eval_total * 100) if eval_total > 0 else 0.0

    print("\n" + "=" * 70)
    print("📊 BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Claims Evaluated : {eval_total} (Skipped/Network Errors: {errors})")
    print(f"Evaluation Duration    : {duration:.2f} seconds (~{duration/total_rows:.2f}s per claim)")
    print(f"Overall Accuracy       : {correct}/{eval_total} = {accuracy:.1f}%")
    print(f"False-VERIFIED Rate    : {false_marked_true}/{false_claims_total} = {false_verified_rate:.1f}%  (CRITICAL SAFETY: Lower is better)")
    print(f"Abstention (Unverified): {unverified_predictions}/{eval_total} = {abstention_rate:.1f}%\n")

    # Confusion Matrix
    labels = ["TRUE", "FALSE", "UNVERIFIABLE"]
    print("Confusion Matrix (Rows: Ground Truth, Cols: Model Predictions):")
    header = " " * 16 + "".join(f"{l:>14}" for l in labels)
    print(header)
    print("-" * len(header))
    for t in labels:
        row_str = f"{t:<16}" + "".join(f"{conf_matrix[(t, p)]:>14}" for p in labels)
        print(row_str)

    print("\nPer-Class Precision & Recall Metrics:")
    for l in labels:
        tp = conf_matrix[(l, l)]
        pred = sum(conf_matrix[(t, l)] for t in labels)
        actual = sum(conf_matrix[(l, p)] for p in labels)
        precision = (tp / pred) if pred else 0.0
        recall = (tp / actual) if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        print(f" - {l:<14}: Precision={precision:.2f} | Recall={recall:.2f} | F1-Score={f1:.2f}")

    print("=" * 70)

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(api_url=url)
