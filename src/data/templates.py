"""
Prompt templates for ACT-ViT datasets.
Standardized for consistent evaluation across models (Llama-3.2, Qwen).
"""

# HotpotQA (No Context)
HOTPOT_NO_CONTEXT = (
    "Question: {question}\n"
    "Answer:"
)

# HotpotQA (With Context)
HOTPOT_WITH_CONTEXT = (
    "Use the context to answer the question.\n\n"
    "Context:\n"
    "{context_blocks}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# TriviaQA
TRIVIA_QA = (
    "Question: {question}\n"
    "Answer:"
)

# Movies (Templated Factual Retrieval)
# Expected input: person, movie
MOVIES_TEMPLATE = "Who acted as {person} in the movie {movie}?"

# IMDB (1-shot)
IMDB_1_SHOT = (
    "Classify the sentiment as exactly one word: Positive or Negative.\n\n"
    "Example:\n"
    "Review: {example_review}\n"
    "Sentiment: Positive\n\n"
    "Now classify:\n"
    "Review: {review}\n"
    "Sentiment:"
)
