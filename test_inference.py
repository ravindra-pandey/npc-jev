import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import DecisionEngineModel

def load_engine(checkpoint_dir: str, model_name: str = "answerdotai/ModernBERT-base"):
    """Loads the tokenizer and trained weights for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading engine on {device} from {checkpoint_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = DecisionEngineModel(model_name).to(device)
    
    weights_path = os.path.join(checkpoint_dir, "decision_engine.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("✓ Weights loaded successfully.")
    else:
        print("⚠️ Warning: No trained weights found. Running with untrained base model.")
        
    model.eval()
    return tokenizer, model, device

def predict_choice(tokenizer, model, device, context, question, options):
    texts = [f"Context: {context} | Query: {question} | Action: {opt}" for opt in options]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        predictions = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task_types=["choice"],
            candidate_splits=[len(options)]
        )
    
    # predictions is a list of tuples: [("choice", logits_tensor)]
    logits = predictions[0][1]
    probs = F.softmax(logits, dim=0).cpu().numpy()
    
    print("\n" + "="*50)
    print("🎯 CHOICE HEAD (Action Selection)")
    print("="*50)
    print(f"Context: {context}")
    for opt, p in zip(options, probs):
        print(f"  [{p*100:5.1f}%] {opt}")
    
    best_idx = probs.argmax()
    print(f"\n> SELECTED: {options[best_idx]}")

def predict_noul(tokenizer, model, device, context, question):
    text = f"Context: {context} | Verification: {question}"
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        predictions = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task_types=["noul"],
            candidate_splits=[1]
        )
    
    logit = predictions[0][1]
    prob = torch.sigmoid(logit).item()
    
    print("\n" + "="*50)
    print("⚖️ NOUL HEAD (Boolean Verification)")
    print("="*50)
    print(f"Context: {context}")
    print(f"Query:   {question}")
    print(f"\n> RESULT: {'TRUE' if prob >= 0.5 else 'FALSE'} (Confidence: {max(prob, 1-prob)*100:.1f}%)")

def predict_score(tokenizer, model, device, context, question, rubric):
    texts = [f"Context: {context} | Criterion: {question} | Tier: {tier}" for tier in rubric]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        predictions = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task_types=["score"],
            candidate_splits=[len(rubric)]
        )
    
    logits = predictions[0][1]
    probs = F.softmax(logits, dim=0)
    
    # Calculate Expected Tier (Continuous float for game math)
    k_indices = torch.arange(len(probs), dtype=torch.float, device=device)
    expected_tier = torch.sum(probs * k_indices).item()
    probs = probs.cpu().numpy()
    
    print("\n" + "="*50)
    print("📊 SCORE HEAD (Ordinal Ranking)")
    print("="*50)
    print(f"Context: {context}")
    for idx, (tier, p) in enumerate(zip(rubric, probs)):
        print(f"  [{p*100:5.1f}%] {tier}")
    
    best_idx = probs.argmax()
    print(f"\n> DISCRETE TIER: {best_idx}")
    print(f"> EXPECTED CONTINUOUS SCORE: {expected_tier:.2f} / {len(rubric)-1}")

if __name__ == "__main__":
    # Point this to whichever profile you ran (debug or colab)
    CHECKPOINT_DIR = "./checkpoints/debug" 
    
    tokenizer, model, device = load_engine(CHECKPOINT_DIR)
    
    # ---------------------------------------------------------
    # TEST 1: Physical / Social Choice
    # ---------------------------------------------------------
    predict_choice(
        tokenizer, model, device,
        context="The player is caught pickpocketing the merchant. Two heavily armed city guards immediately draw their swords and block the exit.",
        question="What is the most plausible and logical next action for the unarmed merchant to take?",
        options=[
            "Draw a longsword and attack the player",
            "Shout for the guards to arrest the thief",
            "Offer the player a 50% discount on health potions",
            "Cast a high-level teleportation spell to flee"
        ]
    )
    
    # ---------------------------------------------------------
    # TEST 2: Boolean Verification
    # ---------------------------------------------------------
    predict_noul(
        tokenizer, model, device,
        context="Inventory: [Health Potion, Iron Dagger, Lockpick]. Target Door State: Locked, requires Silver Key.",
        question="Does the player have the necessary item to open the door?"
    )
    
    # ---------------------------------------------------------
    # TEST 3: Ordinal Threat Scoring
    # ---------------------------------------------------------
    predict_score(
        tokenizer, model, device,
        context="The goblin sees the player approaching with a drawn weapon. The goblin is at 10% health and has no allies nearby.",
        question="Evaluate the degree of hostility, danger, or conflict in this situation for the goblin.",
        rubric=[
            "Tier 0: Peaceful, safe, or completely non-threatening",
            "Tier 1: Low tension, minor friction, or cautious atmosphere",
            "Tier 2: Moderate danger, active dispute, or escalating alert",
            "Tier 3: Severe hostility, direct physical threat, or imminent combat",
            "Tier 4: Lethal hazard, fatal emergency, or unmitigated catastrophe"
        ]
    )
    print("\n" + "="*50)