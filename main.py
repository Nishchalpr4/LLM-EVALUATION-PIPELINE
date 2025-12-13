import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------
# Load utilities
# ----------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_context_texts(context_json):
    texts = []
    for item in context_json.get("data", {}).get("vector_data", []):
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def extract_ai_answer(conversation_json):
    for turn in reversed(conversation_json.get("conversation_turns", [])):
        if turn.get("role") == "AI/Chatbot":
            return turn.get("message", "")
    return ""


def extract_last_user_question(conversation_json):
    for turn in reversed(conversation_json.get("conversation_turns", [])):
        if turn.get("role") == "User":
            return turn.get("message", "")
    return ""


# ----------------------------
# Text processing
# ----------------------------

def split_into_sentences(text):
    if not text:
        return []
    sentences = text.replace("\n", " ").split(".")
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ----------------------------
# Embedding-based hallucination
# ----------------------------

class EmbeddingHallucinationDetector:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def is_claim_supported(self, claim, context_embeddings, threshold=0.60):
        claim_emb = self.encode([claim])
        similarities = cosine_similarity(claim_emb, context_embeddings)[0]
        return np.max(similarities) >= threshold, float(np.max(similarities))

    def detect(self, answer, context_texts, threshold=0.60):
        claims = split_into_sentences(answer)
        if not claims:
            return [], 0.0

        context_embeddings = self.encode(context_texts)

        unsupported = []
        scores = []

        for claim in claims:
            supported, sim = self.is_claim_supported(
                claim, context_embeddings, threshold
            )
            scores.append(sim)
            if not supported:
                unsupported.append(
                    {"claim": claim, "max_similarity": round(sim, 3)}
                )

        hallucination_score = len(unsupported) / len(claims)
        return unsupported, round(hallucination_score, 3)


# ----------------------------
# Main
# ----------------------------

def main():
    conversation = load_json("conversation.json")
    context_json = load_json("context.json")

    context_texts = extract_context_texts(context_json)
    ai_answer = extract_ai_answer(conversation)

    detector = EmbeddingHallucinationDetector()

    unsupported_claims, hallucination_score = detector.detect(
        ai_answer,
        context_texts,
        threshold=0.60  # tune between 0.55–0.70
    )

    print("\nAI Answer:\n")
    print(ai_answer)

    print("\nUnsupported Claims:")
    for u in unsupported_claims:
        print(f"- {u['claim']}  (sim={u['max_similarity']})")

    print("\nHallucination score:", hallucination_score)


if __name__ == "__main__":
    main()
