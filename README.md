# LLM Hallucination Detection Tool

A Python tool that detects hallucinations in LLM-generated responses by comparing them against a context of factual information using embedding-based similarity analysis.

## Overview

This tool evaluates whether claims made by an AI chatbot are supported by the provided context documents. It uses sentence transformers and cosine similarity to detect unsupported or "hallucinated" claims in AI responses.

## Features

- **Embedding-Based Hallucination Detection**: Uses semantic embeddings to compare AI responses against context documents
- **Claim Extraction**: Automatically breaks AI responses into individual claims for analysis
- **Similarity Scoring**: Calculates cosine similarity scores between claims and context
- **Hallucination Scoring**: Generates an overall hallucination score (0.0-1.0) based on unsupported claims

## Project Structure

```
LLM_EVALuation/
├── main.py              # Main entry point with hallucination detection logic
├── conversation.json    # Sample conversation data with AI responses and user messages
├── context.json         # Vector data containing contextual information
└── README.md           # This file
```

## Requirements

- Python 3.7+
- sentence-transformers
- scikit-learn
- numpy

## Installation

Install the required dependencies:

```bash
pip install sentence-transformers scikit-learn numpy
```

## Usage

1. Prepare your data:
   - `conversation.json`: Should contain conversation turns with AI responses
   - `context.json`: Should contain context vector data with relevant background information

2. Run the detector:

```bash
python main.py
```

## How It Works

### 1. Data Loading
The tool extracts:
- Context texts from `context.json` (stored in `data.vector_data[].text`)
- AI responses from `conversation.json` (from turns with role "AI/Chatbot")

### 2. Text Processing
AI responses are split into individual sentences (claims) for analysis. Claims shorter than 10 characters are filtered out.

### 3. Embedding Generation
Uses the `all-MiniLM-L6-v2` sentence transformer model to generate embeddings for:
- Each claim in the AI response
- All context documents

### 4. Similarity Analysis
Compares each claim against context using cosine similarity:
- Finds the maximum similarity score for each claim
- Flags claims with similarity below the threshold (default: 0.60) as unsupported

### 5. Hallucination Scoring
Calculates:
- **Unsupported Claims**: List of claims with low similarity scores
- **Hallucination Score**: Percentage of unsupported claims (unsupported_count / total_claims)

## Configuration

Adjust the similarity threshold in `main.py`:

```python
unsupported_claims, hallucination_score = detector.detect(
    ai_answer,
    context_texts,
    threshold=0.60  # tune between 0.55–0.70
)
```

- **Lower threshold (0.55)**: More lenient, fewer false positives
- **Higher threshold (0.70)**: Stricter, more likely to flag hallucinations

## Output

The tool outputs:
- The complete AI response
- List of unsupported claims with their similarity scores
- Overall hallucination score (0.0 = no hallucinations, 1.0 = all claims unsupported)

Example:
```
AI Answer:
[Full AI response text]

Unsupported Claims:
- Claim 1 text  (sim=0.456)
- Claim 2 text  (sim=0.523)

Hallucination score: 0.234
```

## Classes and Functions

### EmbeddingHallucinationDetector
Main class for detecting hallucinations.

**Methods:**
- `__init__(model_name)`: Initialize with a sentence transformer model
- `encode(texts)`: Generate embeddings for texts
- `is_claim_supported(claim, context_embeddings, threshold)`: Check if a claim is supported
- `detect(answer, context_texts, threshold)`: Detect hallucinations in an answer

### Utility Functions
- `load_json(path)`: Load JSON files
- `extract_context_texts(context_json)`: Extract context from JSON
- `extract_ai_answer(conversation_json)`: Extract AI response
- `extract_last_user_question(conversation_json)`: Extract user query
- `split_into_sentences(text)`: Split text into sentences

## Notes

- The `all-MiniLM-L6-v2` model is automatically downloaded on first use
- Embeddings are normalized for cosine similarity calculations
- Claims shorter than 10 characters are filtered during processing
- The tool is designed for English text

## License

This project is part of an LLM Engineer Internship assignment.
