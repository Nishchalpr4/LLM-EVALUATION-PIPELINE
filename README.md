# LLM Evaluation Pipeline

A lightweight, interpretable evaluation pipeline for **Retrieval-Augmented Generation (RAG)** systems. This project evaluates an AI-generated answer against retrieved context to measure **relevance, hallucination, completeness, estimated cost, and latency**.

---

## 📌 Overview

The pipeline analyzes how well an AI response is grounded in retrieved context from a vector database. It is designed for **baseline or lightweight evaluation**, prioritizing transparency and simplicity over heavy semantic modeling.

### Inputs

* **Conversation log** (`conversation.json`)
* **Retrieved context** from a vector database (`context.json`)

### Output

* Quantitative metrics assessing the quality and efficiency of the AI-generated response.

---

## 📊 Metrics Implemented

### 1. Relevance Score

Measures lexical overlap between the AI-generated answer and the most relevant retrieved context chunk.

### 2. Hallucination Score

Identifies unsupported or ungrounded claims in the AI answer by comparing it against the retrieved context.

### 3. Completeness Score

Estimates how thoroughly the AI response addresses the user’s question.

### 4. Estimated Cost

Approximates evaluation cost using a token-count–based heuristic.

### 5. Latency

Measures total evaluation runtime in **milliseconds**.

---

## 📁 File Structure

```
.
├── conversation.json   # Conversation history (user + AI messages)
├── context.json        # Vector database retrieval output
├── main.py         # Main evaluation script
└── README.md           # Project documentation
```

---

## 🧾 Input Format

### `conversation.json`

```json
{
  "conversation_turns": [
    { "role": "User", "message": "Your question here" },
    { "role": "AI/Chatbot", "message": "AI response here" }
  ]
}
```

### `context.json`

```json
{
  "data": {
    "vector_data": [
      { "text": "Retrieved context chunk 1" },
      { "text": "Retrieved context chunk 2" }
    ]
  }
}
```

---

## ▶️ How to Run

Ensure Python 3 is installed, then execute:

```bash
python evaluate.py
```

---

## 📤 Output

The script prints the following to the console:

* AI-generated answer
* Relevance score
* Unsupported claims (if any)
* Hallucination score
* Completeness score
* Estimated evaluation cost
* Evaluation latency (ms)

---

## ⚠️ Limitations

* Relies on **lexical overlap**, not semantic similarity
* Does not handle paraphrasing or synonymy
* May underestimate relevance for well-written but rephrased answers
* Best suited for **baseline evaluation**, debugging, or early-stage RAG systems

---

## 🚀 Future Improvements

* Replace word overlap with **embedding-based similarity**
* Add **sentence-level semantic grounding**
* Introduce configurable **thresholds and metric weights**
* Add **unit tests** for metric validation
* Support multi-turn conversations and multiple answers

---

## 👤 Author

**Nishchal P R**

---

## 📄 License

This project is intended for educational and experimental use. Add a license file if distributing publicly.
