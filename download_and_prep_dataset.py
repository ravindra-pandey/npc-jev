import json
import random
import re
import os
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = "datasets"
DEBUG_OUT = "debug_sample.jsonl"
VAL_OUT   = "decision_val_3k.jsonl"
TRAIN_OUT = "decision_train_44k.jsonl"

os.makedirs(BASE_DIR, exist_ok=True)
# Quota allocation per source
TARGETS = {
    "choice_hs":   {"debug": 50,  "val": 750,  "train": 11200},  # HellaSwag: 12,000 total (4 candidates)
    "choice_siqa": {"debug": 50,  "val": 750,  "train": 9200},   # Social IQa: 10,000 total (3 candidates)
    "noul_boolq":  {"debug": 40,  "val": 400,  "train": 5560},   # BoolQ: 6,000 total (50/50 balanced)
    "noul_mnli":   {"debug": 60,  "val": 600,  "train": 8340},   # MNLI: 9,000 total (50/50 balanced)
    "score_5tier": {"debug": 50,  "val": 300,  "train": 4650},   # 5-Tier Ordinal: 5,000 total (Tiers 0..4)
    "score_3tier": {"debug": 50,  "val": 300,  "train": 4650}    # 3-Tier Ordinal: 5,000 total (Tiers 0..2)
}

TOTAL_NEED = {k: sum(TARGETS[k].values()) for k in TARGETS}

# ---------------------------------------------------------------------------
# 1. CHOICE: HellaSwag (Physical Commonsense & Action Evaluation)
# ---------------------------------------------------------------------------
def fetch_hellaswag(needed):
    print(f"\n[1/6] Streaming {needed} physical action choices from HellaSwag...")
    stream = load_dataset("Rowan/hellaswag", split="train", streaming=True)
    records = []
    pbar = tqdm(total=needed, desc="HellaSwag")

    for row in stream:
        endings = [str(e).strip() for e in row.get("endings", [])]
        if len(endings) != 4 or any(len(e) == 0 for e in endings):
            continue
        try:
            target = int(row.get("label", -1))
        except (ValueError, TypeError):
            continue
        if target not in [0, 1, 2, 3]:
            continue

        ctx = f"{row.get('activity_label', '')}: {row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip()
        records.append({
            "id": f"hs_{len(records)}",
            "domain": "physical_action",
            "kind": "choice",
            "context": ctx,
            "question": "What is the most plausible and logical next action to take?",
            "options": endings,
            "target": target
        })
        pbar.update(1)
        if len(records) >= needed:
            break
    pbar.close()
    return records

# ---------------------------------------------------------------------------
# 2. CHOICE: Social IQa (Human Social Intent & Dialogue Reactions)
# ---------------------------------------------------------------------------
def fetch_social_iqa(needed):
    print(f"\n[2/6] Streaming {needed} social interaction choices from Social IQa...")
    try:
        stream = load_dataset("tasksource/social_i_qa", split="train", streaming=True)
    except Exception as e:
        print(f"Warning: Primary Social IQa mirror error ({e}). Using direct fallback...")
        stream = load_dataset("allenai/social_i_qa", split="train", streaming=True)

    records = []
    pbar = tqdm(total=needed, desc="Social IQa")

    for row in stream:
        opts = [
            str(row.get("answerA", "")).strip(),
            str(row.get("answerB", "")).strip(),
            str(row.get("answerC", "")).strip()
        ]
        if any(len(o) == 0 for o in opts):
            continue
        try:
            raw_label = row.get("label")
            target = int(raw_label) - 1  # Convert 1-indexed to 0-indexed
        except (ValueError, TypeError):
            continue
            
        if target not in [0, 1, 2]:
            continue

        records.append({
            "id": f"siqa_{len(records)}",
            "domain": "social_intent",
            "kind": "choice",
            "context": str(row.get("context", "")).strip(),
            "question": str(row.get("question", "")).strip(),
            "options": opts,
            "target": target
        })
        pbar.update(1)
        if len(records) >= needed:
            break
    pbar.close()
    return records

# ---------------------------------------------------------------------------
# 3. NOUL: Google BoolQ (Strictly Balanced: 3,000 True / 3,000 False)
# ---------------------------------------------------------------------------
def fetch_boolq(needed):
    print(f"\n[3/6] Streaming {needed} verified boolean records from BoolQ (50/50 balanced)...")
    stream = load_dataset("google/boolq", split="train", streaming=True)
    half = needed // 2
    true_pool, false_pool = [], []
    pbar = tqdm(total=needed, desc="BoolQ")

    for row in stream:
        ans = row.get("answer")
        if ans is None:
            continue
        target_val = 1.0 if ans is True else 0.0

        rec = {
            "id": f"boolq_{len(true_pool) + len(false_pool)}",
            "domain": "reading_verification",
            "kind": "noul",
            "context": str(row.get("passage", "")).strip(),
            "question": f"Is this state or rule verified to be true: \"{row.get('question', '').strip()}\"?",
            "options": ["false", "true"],
            "target": target_val
        }

        if target_val == 1.0 and len(true_pool) < half:
            true_pool.append(rec)
            pbar.update(1)
        elif target_val == 0.0 and len(false_pool) < (needed - half):
            false_pool.append(rec)
            pbar.update(1)

        if len(true_pool) >= half and len(false_pool) >= (needed - half):
            break
    pbar.close()
    combined = true_pool + false_pool
    random.seed(42); random.shuffle(combined)
    return combined

# ---------------------------------------------------------------------------
# 4. NOUL: MultiNLI (Deductive Logic: Exactly 4,500 True / 4,500 False)
# ---------------------------------------------------------------------------
def fetch_mnli(needed):
    print(f"\n[4/6] Streaming {needed} deductive logic verification records from MNLI...")
    stream = load_dataset("nyu-mll/multi_nli", split="train", streaming=True)
    half = needed // 2
    entail_pool, contra_pool = [], []
    pbar = tqdm(total=needed, desc="MNLI Logic")

    for row in stream:
        label = row.get("label")
        if label not in [0, 2]:  # 0: entailment (true), 2: contradiction (false)
            continue

        target_val = 1.0 if label == 0 else 0.0
        premise = str(row.get("premise", "")).strip()
        hypothesis = str(row.get("hypothesis", "")).strip()

        rec = {
            "id": f"mnli_{len(entail_pool) + len(contra_pool)}",
            "domain": "deductive_logic",
            "kind": "noul",
            "context": f"Observed Premise: {premise}",
            "question": f"Based strictly on the premise, is this statement necessarily true: \"{hypothesis}\"?",
            "options": ["false", "true"],
            "target": target_val
        }

        if target_val == 1.0 and len(entail_pool) < half:
            entail_pool.append(rec)
            pbar.update(1)
        elif target_val == 0.0 and len(contra_pool) < (needed - half):
            contra_pool.append(rec)
            pbar.update(1)

        if len(entail_pool) >= half and len(contra_pool) >= (needed - half):
            break
    pbar.close()
    combined = entail_pool + contra_pool
    random.seed(42); random.shuffle(combined)
    return combined

# ---------------------------------------------------------------------------
# 5. SCORE: 5-Tier Ordinal Scales (Varied Rubrics & Questions, Tiers 0..4)
# ---------------------------------------------------------------------------
def fetch_score_5tier(needed):
    print(f"\n[5/6] Streaming {needed} 5-tier ordinal records with varied rubric phrasings...")
    stream = load_dataset("SetFit/amazon_reviews_multi_en", split="train", streaming=True)

    # 4 distinct rubric definitions and questions to prevent shortcut token learning
    RUBRIC_5_VARIANTS = [
        {
            "question": "Assess the state and select the appropriate satisfaction or quality tier.",
            "options": [
                "1 star: Highly critical, unacceptable, or extremely negative evaluation",
                "2 stars: Substandard, unsatisfactory, with notable defects or concerns",
                "3 stars: Neutral, mediocre, or average meeting basic expectations",
                "4 stars: Favorable, solid performance, and generally positive satisfaction",
                "5 stars: Outstanding, flawless excellence, and highest commendation"
            ]
        },
        {
            "question": "Evaluate the situation and assign the corresponding intensity level from 0 to 4.",
            "options": [
                "Tier 0: Negligible severity, completely peaceful or routine",
                "Tier 1: Low impact, minor friction or slight concern",
                "Tier 2: Moderate impact, noticeable tension requiring awareness",
                "Tier 3: High impact, severe urgency or acute escalation",
                "Tier 4: Critical impact, maximum hazard or immediate emergency"
            ]
        },
        {
            "question": "Rate the operational priority requiring intervention according to the rubric.",
            "options": [
                "Priority 0: Routine background status, no response needed",
                "Priority 1: Low priority, deferrable item with trivial consequence",
                "Priority 2: Medium priority, standard queue processing",
                "Priority 3: High priority, expedited resolution requested",
                "Priority 4: Emergency priority, blocking crisis requiring immediate action"
            ]
        },
        {
            "question": "Determine the subject's stance or willingness to cooperate on a 5-point scale.",
            "options": [
                "Level 0: Total refusal, hostile stance, or deep resistance",
                "Level 1: Reluctant, cautious, or skeptical disposition",
                "Level 2: Neutral, transactional, or indifferent posture",
                "Level 3: Receptive, cooperative, or favorable attitude",
                "Level 4: Completely aligned, eager partner, or enthusiastic ally"
            ]
        }
    ]

    tier_bins = defaultdict(list)
    per_tier = needed // 5
    pbar = tqdm(total=needed, desc="5-Tier Score")

    for row in stream:
        try:
            tier = int(row.get("label", -1))
        except (ValueError, TypeError):
            continue
        if tier not in [0, 1, 2, 3, 4]:
            continue

        if len(tier_bins[tier]) < per_tier:
            text = str(row.get("text", "")).strip().replace("\n", " ")
            if len(text) > 400:
                text = text[:400] + "..."

            # Alternate rubric phrasing across samples
            rubric = RUBRIC_5_VARIANTS[len(tier_bins[tier]) % len(RUBRIC_5_VARIANTS)]

            tier_bins[tier].append({
                "id": f"score_5t_{tier}_{len(tier_bins[tier])}",
                "domain": "ordinal_5tier",
                "kind": "score",
                "context": text,
                "question": rubric["question"],
                "options": rubric["options"],
                "target": tier
            })
            pbar.update(1)

        if all(len(tier_bins[t]) >= per_tier for t in range(5)):
            break
    pbar.close()

    combined = []
    for t in range(5):
        combined.extend(tier_bins[t][:per_tier])
    random.seed(42); random.shuffle(combined)
    return combined

# ---------------------------------------------------------------------------
# 6. SCORE: 3-Tier Ordinal Scales (Varied Rubrics & Questions, Tiers 0..2)
# ---------------------------------------------------------------------------
def fetch_score_3tier(needed):
    print(f"\n[6/6] Streaming {needed} 3-tier ordinal records with varied rubric phrasings...")
    stream = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train", streaming=True)

    RUBRIC_3_VARIANTS = [
        {
            "question": "Assess the expressed demeanor and classify the emotional valence tier.",
            "options": [
                "Negative valence: Adverse, frustrated, hostile, or displeased reaction",
                "Neutral valence: Objective, detached, transactional, or indifferent posture",
                "Positive valence: Friendly, pleased, supportive, or cooperative demeanor"
            ]
        },
        {
            "question": "Classify the observed tension level based on the context.",
            "options": [
                "Tier 0: Friction or conflict observed",
                "Tier 1: Calm, balanced, or non-reactive state",
                "Tier 2: Constructive, encouraging, or welcoming energy"
            ]
        },
        {
            "question": "Determine the threat posture indicated in this communication.",
            "options": [
                "Low threat: Mild irritation or non-threatening stance",
                "Moderate threat: Guarded or defensive stance",
                "High threat: Escalated hostility or explicit confrontation"
            ]
        }
    ]

    tier_bins = defaultdict(list)
    per_tier = needed // 3
    pbar = tqdm(total=needed, desc="3-Tier Score")

    for row in stream:
        try:
            tier = int(row.get("label", -1))
        except (ValueError, TypeError):
            continue
        if tier not in [0, 1, 2]:
            continue

        if len(tier_bins[tier]) < per_tier:
            text = str(row.get("text", "")).strip().replace("\n", " ")
            rubric = RUBRIC_3_VARIANTS[len(tier_bins[tier]) % len(RUBRIC_3_VARIANTS)]

            tier_bins[tier].append({
                "id": f"score_3t_{tier}_{len(tier_bins[tier])}",
                "domain": "ordinal_3tier",
                "kind": "score",
                "context": text,
                "question": rubric["question"],
                "options": rubric["options"],
                "target": tier
            })
            pbar.update(1)

        if all(len(tier_bins[t]) >= per_tier for t in range(3)):
            break
    pbar.close()

    combined = []
    for t in range(3):
        combined.extend(tier_bins[t][:per_tier])
    random.seed(42); random.shuffle(combined)
    return combined

# ---------------------------------------------------------------------------
# MAIN ASSEMBLER & NON-OVERLAPPING SPLITTER
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("EXECUTING RIGOROUS MULTI-TASK DECISION DATASET ENGINE")
    print("=" * 80)

    pools = {
        "choice_hs":   fetch_hellaswag(TOTAL_NEED["choice_hs"]),
        "choice_siqa": fetch_social_iqa(TOTAL_NEED["choice_siqa"]),
        "noul_boolq":  fetch_boolq(TOTAL_NEED["noul_boolq"]),
        "noul_mnli":   fetch_mnli(TOTAL_NEED["noul_mnli"]),
        "score_5tier": fetch_score_5tier(TOTAL_NEED["score_5tier"]),
        "score_3tier": fetch_score_3tier(TOTAL_NEED["score_3tier"])
    }

    debug_pool, val_pool, train_pool = [], [], []

    for key, data in pools.items():
        d_cnt = TARGETS[key]["debug"]
        v_cnt = TARGETS[key]["val"]
        t_cnt = TARGETS[key]["train"]

        # Exact, mutually exclusive slicing
        debug_pool.extend(data[:d_cnt])
        val_pool.extend(data[d_cnt : d_cnt + v_cnt])
        train_pool.extend(data[d_cnt + v_cnt : d_cnt + v_cnt + t_cnt])

    random.seed(42); random.shuffle(debug_pool)
    random.seed(42); random.shuffle(val_pool)
    random.seed(42); random.shuffle(train_pool)

    with open(f"{BASE_DIR}/{DEBUG_OUT}", "w", encoding="utf-8") as f:
        for r in debug_pool: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ Saved Debug Set:      {DEBUG_OUT} ({len(debug_pool):,} rows)")

    with open(f"{BASE_DIR}/{VAL_OUT}", "w", encoding="utf-8") as f:
        for r in val_pool: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ Saved Validation Set: {VAL_OUT} ({len(val_pool):,} rows)")

    with open(f"{BASE_DIR}/{TRAIN_OUT}", "w", encoding="utf-8") as f:
        for r in train_pool: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ Saved Training Set:   {TRAIN_OUT} ({len(train_pool):,} rows)")
    print("=" * 80)

if __name__ == "__main__":
    main()