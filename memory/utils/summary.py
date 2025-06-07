# memory/utils/summary.py
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_chunks(chunks: list[str], max_tokens=300) -> str:
    combined = " ".join(chunks)
    summary = summarizer(combined, max_length=max_tokens, min_length=50, do_sample=False)
    return summary[0]['summary_text']
