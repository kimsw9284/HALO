import json
import os
import glob
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from litellm import completion
from pypdf import PdfReader

MODEL = "openai/gpt-oss-120b"
API_BASE = "http://localhost:4000"
#API_KEY = "anything"

DOCS_FOLDER = "docs"
CHUNK_SIZE = 300

# =====================================================================
# TOOL DEFINITIONS (API + RAG Search)
# =====================================================================

def fetch_hgcal_part_full(barcode: str) -> str:
    """Queries the HGCAL database via HGCAPI for full part details."""
    url = f"https://hgcapi-cmsr.web.cern.ch/part/{barcode}/full"
    headers = {"accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        return f"HGCAPI Error: Received status code {response.status_code} for barcode {barcode}."
    except Exception as e:
        return f"Failed to connect to HGCAPI: {str(e)}"


# Define tools for LiteLLM tool calling format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_hgcal_part_full",
            "description": "Fetch complete records and metadata for an HGCAL part using its barcode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "barcode": {
                        "type": "string",
                        "description": "The exact barcode of the HGCAL part (e.g., '320EH0QH0010012')",
                    }
                },
                "required": ["barcode"],
            },
        },
    }
]

# =====================================================================
# RAG PIPELINE (LOCAL DOCS)
# =====================================================================

def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def load_and_chunk_docs(folder):
    chunks = []
    if not os.path.exists(folder):
        return chunks

    for filepath in glob.glob(os.path.join(folder, "*.txt")):
        with open(filepath, "r") as f:
            text = f.read()
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if chunk:
                chunks.append({"source": os.path.basename(filepath), "text": chunk})

    for filepath in glob.glob(os.path.join(folder, "*.pdf")):
        text = extract_text_from_pdf(filepath)
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if chunk:
                chunks.append({"source": os.path.basename(filepath), "text": chunk})

    return chunks

def build_index(chunks):
    if not chunks:
        return None, None
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def retrieve(question, chunks, vectorizer, matrix, top_k=2):
    if not vectorizer or matrix is None:
        return []
    question_vec = vectorizer.transform([question])
    scores = cosine_similarity(question_vec, matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# =====================================================================
# AGENT REASONING LOOP
# =====================================================================

def run_agent(user_query, chunks, vectorizer, matrix):
    # Step 1: Retrieve local docs if available
    local_chunks = retrieve(user_query, chunks, vectorizer, matrix) if chunks else []
    
    system_prompt = (
        "You are an assistant for the CMS HGCAL project. You have access to tools "
        "to query the live HGCAL database (HGCAPI) by barcode, as well as local context files.\n"
    )
    if local_chunks:
        doc_context = "\n\n".join([f"[{c['source']}] {c['text']}" for c in local_chunks])
        system_prompt += f"\nLOCAL DOCUMENT CONTEXT:\n{doc_context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    # Step 2: First LLM call — let LLM decide if it needs to execute a tool
    response = completion(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        api_base=API_BASE,
        api_key=API_KEY,
    )

    response_message = response.choices[0].message

    # Step 3: Handle Tool Calling execution
    if getattr(response_message, "tool_calls", None):
        tool_call = response_message.tool_calls[0]
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if func_name == "fetch_hgcal_part_full":
            barcode = args.get("barcode")
            print(f"\n[Agent Tool Call] Fetching data for barcode '{barcode}' from HGCAPI...")
            
            # Execute tool
            api_result = fetch_hgcal_part_full(barcode)

            # Append assistant message and tool response message back into chat history
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": api_result,
            })

            # Step 4: Final LLM call — synthesize live API payload into plain English
            final_response = completion(
                model=MODEL,
                messages=messages,
                api_base=API_BASE,
                api_key=API_KEY,
            )
            return final_response.choices[0].message.content

    return response_message.content

# =====================================================================
# ENTRY POINT
# =====================================================================

def main():
    print("Loading local documents...")
    chunks = load_and_chunk_docs(DOCS_FOLDER)
    vectorizer, matrix = build_index(chunks)
    print(f"Agent initialized. Ready to query local docs or HGCAPI.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break

        answer = run_agent(question, chunks, vectorizer, matrix)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
