RAG Evaluation Pipeline

This project implements a lightweight evaluation pipeline for Retrieval-Augmented Generation (RAG) systems. It analyzes an AI response against retrieved context to measure relevance, hallucination, completeness, cost, and latency.

Overview

The pipeline takes:

A conversation log (conversation.json)

Retrieved context from a vector database (context.json)

It produces quantitative metrics to assess the quality of the AI-generated answer.

Metrics Implemented

Relevance Score
Measures lexical overlap between the AI answer and the most relevant context chunk.

Hallucination Score
Identifies unsupported claims in the AI answer based on context grounding.

Completeness Score
Estimates how much of the user question is addressed in the AI response.

Estimated Cost
Approximates evaluation cost using a token-based heuristic.

Latency
Measures total evaluation runtime in milliseconds.

File Structure
.
├── conversation.json   # Conversation history (user + AI messages)
├── context.json        # Vector database retrieval output
├── evaluate.py         # Main evaluation script
└── README.md

Input Format
conversation.json
{
  "conversation_turns": [
    { "role": "User", "message": "Your question here" },
    { "role": "AI/Chatbot", "message": "AI response here" }
  ]
}

context.json
{
  "data": {
    "vector_data": [
      { "text": "Retrieved context chunk 1" },
      { "text": "Retrieved context chunk 2" }
    ]
  }
}

How to Run
python evaluate.py

Output

The script prints:

AI answer

Relevance score

Unsupported claims (if any)

Hallucination score

Completeness score

Estimated evaluation cost

Evaluation latency

Limitations

Uses lexical overlap, not semantic similarity

Does not handle paraphrases or synonyms

Best suited for lightweight or baseline evaluation

Future Improvements

Replace word overlap with embedding-based similarity

Add sentence-level semantic grounding

Introduce configurable thresholds and weights

Add unit tests for metric validation

Author

Nischal P R
