import json
import time
import re


# ==================== I/O ====================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================== NORMALIZATION ====================

def normalize_text(text):
    """
    Lowercase, remove URLs and punctuation for robust matching
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==================== EXTRACTION ====================

def extract_context_texts(context_json):
    """
    Extract text chunks from vector DB response
    """
    texts = []

    if "data" in context_json and "vector_data" in context_json["data"]:
        for item in context_json["data"]["vector_data"]:
            if "text" in item and isinstance(item["text"], str):
                texts.append(item["text"])

    return texts


def extract_ai_answer(conversation_json):
    """
    Extract the last AI/Chatbot message
    """
    turns = conversation_json.get("conversation_turns", [])

    for turn in reversed(turns):
        if turn.get("role") == "AI/Chatbot":
            return turn.get("message", "")

    return ""


def extract_last_user_question(conversation_json):
    """
    Extract the last User message
    """
    turns = conversation_json.get("conversation_turns", [])

    for turn in reversed(turns):
        if turn.get("role") == "User":
            return turn.get("message", "")

    return ""


# ==================== RELEVANCE ====================

def filter_relevant_contexts(question, context_texts, min_overlap=2):
    """
    Filter context chunks using lexical overlap with user question
    """
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
    """
    Word-overlap relevance score
    """
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


# ==================== HALLUCINATION ====================

def split_into_sentences(text):
    """
    Naive sentence splitter
    """
    if not text:
        return []

    sentences = text.split(".")
    return [s.strip() for s in sentences if s.strip()]


def is_claim_supported(claim, context_texts, min_overlap=1):
    """
    Check whether claim is supported by any context chunk
    """
    claim_words = set(normalize_text(claim).split())
    if not claim_words:
        return True

    for ctx in context_texts:
        ctx_words = set(normalize_text(ctx).split())
        if len(claim_words & ctx_words) >= min_overlap:
            return True

    return False


def detect_hallucinations(answer, context_texts):
    """
    Returns unsupported claims and hallucination score
    """
    sentences = split_into_sentences(answer)
    unsupported = []

    for sentence in sentences:
        # ignore trivial fragments
        if len(sentence.split()) < 5:
            continue

        if not is_claim_supported(sentence, context_texts):
            unsupported.append(sentence)

    if not sentences:
        score = 0.0
    else:
        score = len(unsupported) / len(sentences)

    return unsupported, score


# ==================== COMPLETENESS ====================

def completeness_score(user_question, answer):
    """
    Measures how much of the user question is covered by the answer
    """
    if not user_question or not answer:
        return 0.0

    q_words = set(normalize_text(user_question).split())
    a_words = set(normalize_text(answer).split())

    if not q_words:
        return 1.0

    overlap = len(q_words & a_words)
    return overlap / len(q_words)


# ==================== COST ====================

def estimate_cost(answer, context_texts, cost_per_1k_tokens=0.001):
    """
    Rough cost estimate based on token count
    """
    total_text = normalize_text(answer) + " ".join(
        normalize_text(ctx) for ctx in context_texts
    )
    token_count = len(total_text.split())
    return (token_count / 1000) * cost_per_1k_tokens


# ==================== MAIN PIPELINE ====================

def main():
    start_time = time.time()

    conversation = load_json("conversation.json")
    context_json = load_json("context.json")

    # Extraction
    context_texts = extract_context_texts(context_json)
    ai_answer = extract_ai_answer(conversation)
    user_question = extract_last_user_question(conversation)

    # Relevance
    filtered_contexts = filter_relevant_contexts(
        user_question, context_texts
    )
    relevance = relevance_score(ai_answer, filtered_contexts)

    # Hallucination
    unsupported_claims, hallucination_score = detect_hallucinations(
        ai_answer, filtered_contexts
    )

    # Completeness
    completeness = completeness_score(user_question, ai_answer)

    # Cost & Latency
    estimated_cost = estimate_cost(ai_answer, filtered_contexts)
    latency_ms = (time.time() - start_time) * 1000

    # Output
    print("AI Answer:")
    print(ai_answer)

    print("\nRelevance score:", round(relevance, 3))

    print("\nUnsupported claims:")
    if unsupported_claims:
        for claim in unsupported_claims:
            print("-", claim)
    else:
        print("None")

    print("\nHallucination score:", round(hallucination_score, 3))
    print("Completeness score:", round(completeness, 3))
    print("Estimated evaluation cost ($):", round(estimated_cost, 6))
    print("Evaluation latency (ms):", round(latency_ms, 2))


if __name__ == "__main__":
    main()