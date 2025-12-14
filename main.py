import json
import time
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==================== I/O ====================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================== NORMALIZATION ====================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==================== EXTRACTION ====================

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


# ==================== RELEVANCE ====================

def filter_relevant_contexts(question, context_texts, min_overlap=2):
    if not question:
        return context_texts

    q_words = set(normalize_text(question).split())
    filtered = []

    for ctx in context_texts:
        ctx_words = set(normalize_text(ctx).split())
        if len(q_words & ctx_words) >= min_overlap:
            filtered.append(ctx)

    return filtered


def relevance_score(answer, context_texts):
    if not answer:
        return 0.0

    answer_words = set(normalize_text(answer).split())
    if not answer_words:
        return 0.0

    max_overlap = 0
    for ctx in context_texts:
        ctx_words = set(normalize_text(ctx).split())
        overlap = len(answer_words & ctx_words)
        max_overlap = max(max_overlap, overlap)

    return max_overlap / len(answer_words)


# ==================== HALLUCINATION (EMBEDDINGS) ====================

def split_into_sentences(text):
    if not text:
        return []
    sentences = text.replace("\n", " ").split(".")
    return [s.strip() for s in sentences if len(s.strip().split()) >= 5]


class EmbeddingHallucinationDetector:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def detect(self, answer, context_texts, threshold=0.60):
        claims = split_into_sentences(answer)
        if not claims or not context_texts:
            return [], 0.0

        context_embeddings = self.encode(context_texts)

        unsupported = []

        for claim in claims:
            claim_emb = self.encode([claim])
            similarities = cosine_similarity(claim_emb, context_embeddings)[0]
            max_sim = float(np.max(similarities))

            if max_sim < threshold:
                unsupported.append({
                    "claim": claim,
                    "max_similarity": round(max_sim, 3)
                })

        hallucination_score = len(unsupported) / len(claims)
        return unsupported, round(hallucination_score, 3)


# ==================== COMPLETENESS ====================

def completeness_score(user_question, answer):
    if not user_question or not answer:
        return 0.0

    q_words = set(normalize_text(user_question).split())
    a_words = set(normalize_text(answer).split())

    if not q_words:
        return 1.0

    return len(q_words & a_words) / len(q_words)


# ==================== COST ====================

def estimate_cost(answer, context_texts, cost_per_1k_tokens=0.001):
    combined_text = normalize_text(answer) + " ".join(
        normalize_text(ctx) for ctx in context_texts
    )
    token_count = len(combined_text.split())
    return (token_count / 1000) * cost_per_1k_tokens


# ==================== MAIN PIPELINE ====================

def main():
    start_time = time.time()

    conversation = load_json("conversation.json")
    context_json = load_json("context.json")

    context_texts = extract_context_texts(context_json)
    ai_answer = extract_ai_answer(conversation)
    user_question = extract_last_user_question(conversation)

    filtered_contexts = filter_relevant_contexts(
        user_question, context_texts
    )

    relevance = relevance_score(ai_answer, filtered_contexts)

    detector = EmbeddingHallucinationDetector()
    unsupported_claims, hallucination_score = detector.detect(
        ai_answer,
        filtered_contexts,
        threshold=0.60  # tune 0.55–0.70
    )

    completeness = completeness_score(user_question, ai_answer)
    estimated_cost = estimate_cost(ai_answer, filtered_contexts)
    latency_ms = (time.time() - start_time) * 1000

    print("\nAI Answer:\n")
    print(ai_answer)

    print("\nRelevance score:", round(relevance, 3))

    print("\nUnsupported claims:")
    if unsupported_claims:
        for u in unsupported_claims:
            print(f"- {u['claim']} (sim={u['max_similarity']})")
    else:
        print("None")

    print("\nHallucination score:", hallucination_score)
    print("Completeness score:", round(completeness, 3))
    print("Estimated evaluation cost ($):", round(estimated_cost, 6))
    print("Evaluation latency (ms):", round(latency_ms, 2))


if __name__ == "__main__":
    main()