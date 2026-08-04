"""
Minimal RAG (Retrieval-Augmented Generation) demo.

Goal: show how to make an LLM answer ONLY using specific documents you
provide, instead of pulling from everything it learned in training.

Pipeline:
  1. Load all .txt files from docs/
  2. Split them into small chunks
  3. Turn chunks into vectors using TF-IDF (a simple, free, offline
     way to represent text as numbers based on word frequency)
  4. When the user asks a question, turn the question into a vector too,
     and find the chunks whose vectors are most similar (cosine similarity)
  5. Stuff only those chunks into the prompt, and tell the LLM (via litellm)
     to answer using ONLY that context
  6. Send it to the model and print the answer

This uses TF-IDF instead of a real embeddings API so you can run the
retrieval step completely for free, offline, no API key needed.
litellm is only used for the final step: generating the answer.

Install first:
    pip install litellm scikit-learn --break-system-packages
"""

import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from litellm import completion
from pypdf import PdfReader

# ---------------------------------------------------------------------
# STEP 1: Model + proxy setup
# ---------------------------------------------------------------------
# This is set up to talk to a local litellm proxy (e.g. running on
# localhost:4000) serving gpt-oss-120b, matching this curl call:
#
#   curl http://localhost:4000/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model": "gpt-oss-120b", "messages": [...]}'
#
# The "openai/" prefix tells litellm to speak the OpenAI-compatible
# wire format, which is what your proxy expects.
MODEL = "openai/gpt-oss-120b"
API_BASE = "http://localhost:4000"
API_KEY = "anything"  # placeholder — only matters if your proxy enforces auth

DOCS_FOLDER = "docs"
CHUNK_SIZE = 300  # characters per chunk — small so retrieval is precise


def extract_text_from_pdf(filepath):
    """Pull all text out of a PDF, page by page."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_and_chunk_docs(folder):
    """Read every .txt and .pdf file in the folder and split into small chunks."""
    chunks = []

    # plain text files (kept for the earlier demo docs)
    for filepath in glob.glob(os.path.join(folder, "*.txt")):
        with open(filepath, "r") as f:
            text = f.read()
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if chunk:
                chunks.append({"source": os.path.basename(filepath), "text": chunk})

    # PDF files
    for filepath in glob.glob(os.path.join(folder, "*.pdf")):
        text = extract_text_from_pdf(filepath)
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if chunk:
                chunks.append({"source": os.path.basename(filepath), "text": chunk})

    return chunks


def build_index(chunks):
    """Turn all chunks into TF-IDF vectors we can compare against."""
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve(question, chunks, vectorizer, matrix, top_k=2):
    """Find the top_k chunks most relevant to the question."""
    question_vec = vectorizer.transform([question])
    scores = cosine_similarity(question_vec, matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def ask(question, context_chunks):
    """Send the question + retrieved context to the LLM via litellm."""
    context_text = "\n\n".join(
        f"[From {c['source']}]\n{c['text']}" for c in context_chunks
    )

    system_prompt = (
        "You are a helpful assistant that answers ONLY using the provided "
        "context below. If the answer is not contained in the context, "
        "say you don't have that information — do not guess or use "
        "outside knowledge.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        api_base=API_BASE,
        api_key=API_KEY,
    )
    return response.choices[0].message.content


def main():
    print("Loading and indexing documents...")
    chunks = load_and_chunk_docs(DOCS_FOLDER)
    vectorizer, matrix = build_index(chunks)
    print(f"Indexed {len(chunks)} chunks from {DOCS_FOLDER}/\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break

        top_chunks = retrieve(question, chunks, vectorizer, matrix)
        print("\n--- Retrieved context ---")
        for c in top_chunks:
            print(f"[{c['source']}] {c['text'][:80]}...")
        print("--------------------------\n")

        answer = ask(question, top_chunks)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
