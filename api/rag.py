import os
import json
import requests
import faiss
import numpy as np
import base64
import mimetypes
import sys
import gzip # Import gzip for loading the downloaded .gz file
from dotenv import load_dotenv

load_dotenv()

# --- API Config ---
AI_PROXY_BASE = os.getenv("AI_PROXY_BASE", "https://aipipe.org/openai/v1")
EMBED_ENDPOINT = f"{AI_PROXY_BASE}/embeddings"
CHAT_ENDPOINT = f"{AI_PROXY_BASE}/chat/completions"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("AI_PROXY_API_KEY")
if not API_KEY:
    raise ValueError("AI_PROXY_API_KEY missing")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# --- File Paths ---
BASE_DIR = os.path.dirname(__file__)
INDEX_FILE = os.path.join(BASE_DIR, "faiss_index", "index.faiss")
METADATA_FILE = os.path.join(BASE_DIR, "faiss_index", "metadatas.json.gz") # This file will be downloaded

# --- Google Drive Config ---
# Ensure GDRIVE_METADATA_ID is set in Render Environment Variables
GDRIVE_FILE_ID = os.getenv("GDRIVE_METADATA_ID")
if not GDRIVE_FILE_ID:
    print("WARNING: GDRIVE_METADATA_ID environment variable not set. Metadata will not be downloaded.", file=sys.stderr)


# --- Global Lazy Variables ---
_lazy_index = None
_lazy_metadata = None


def download_metadata_from_gdrive():
    """
    Downloads the metadata.json.gz file from Google Drive.
    """
    if not GDRIVE_FILE_ID:
        print("Error: GDRIVE_METADATA_ID is not set. Cannot download metadata.", file=sys.stderr)
        return False

    print(f"🔹 Downloading metadata from Google Drive (ID: {GDRIVE_FILE_ID})...")
    url = f"https://docs.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    
    try:
        response = requests.get(url, stream=True, timeout=300) # Increased timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
        with open(METADATA_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Download complete.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading metadata from Google Drive: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during Google Drive download: {e}", file=sys.stderr)
        return False


def load_index_and_metadata():
    global _lazy_index, _lazy_metadata
    if _lazy_index and _lazy_metadata:
        return _lazy_index, _lazy_metadata

    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"{INDEX_FILE} missing. Ensure it is committed to Git.")
    
    # Download metadata if it doesn't exist locally (for Render deployment)
    if not os.path.exists(METADATA_FILE):
        if not download_metadata_from_gdrive():
            raise FileNotFoundError(f"{METADATA_FILE} could not be downloaded or is missing.")

    print("🔹 Loading FAISS index and metadata")
    _lazy_index = faiss.read_index(INDEX_FILE)
    with gzip.open(METADATA_FILE, "rt", encoding="utf-8") as f:
        _lazy_metadata = json.load(f)
    return _lazy_index, _lazy_metadata


def get_embedding(text):
    """
    Fetches a single embedding for the given text from the embedding endpoint.
    Handles API errors and retries.
    """
    try:
        resp = requests.post(EMBED_ENDPOINT, json={"model": EMBED_MODEL, "input": [text]}, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return np.array(resp.json()["data"][0]["embedding"]).astype("float32")
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding from {EMBED_ENDPOINT}: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}", file=sys.stderr)
            print(f"Response body: {e.response.text}", file=sys.stderr)
        raise

def cosine_sim(a, b):
    """
    Calculates cosine similarity between two vectors.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def get_post_number_from_url(url):
    """
    Extracts the Discourse post number from a URL, if applicable.
    """
    try:
        if "discourse.onlinedegree.iitm.ac.in/t/" in url:
            parts = url.rstrip("/").split("/")
            if parts and parts[-1].isdigit():
                return int(parts[-1])
        return -1
    except ValueError:
        return -1

def format_discourse_url(doc):
    """
    Formats Discourse URLs for consistency.
    """
    url = doc.get("url", "")
    topic_id = doc.get("topic_id")
    
    if "discourse.onlinedegree.iitm.ac.in/t/" in url and topic_id is not None:
        parts = url.rstrip("/").split("/")
        post_number_str = parts[-1] if parts else ""
        if not post_number_str.isdigit():
            return url

        cleaned_parts = []
        found_t = False
        for part in parts:
            cleaned_parts.append(part)
            if part == "t" and not found_t:
                found_t = True
            elif found_t and part.isdigit() and int(part) == topic_id and len(cleaned_parts) > 1 and cleaned_parts[-2] != "t":
                cleaned_parts.pop(-1)
        
        if len(cleaned_parts) > 2 and cleaned_parts[-2].isdigit() and cleaned_parts[-1].isdigit():
            pass
        elif len(cleaned_parts) > 1 and cleaned_parts[-1].isdigit() and not cleaned_parts[-2].isdigit():
            cleaned_parts.insert(len(cleaned_parts) - 1, str(topic_id))

        return "/".join(cleaned_parts)

    return url

def clean_content_for_prompt(content):
    """
    Cleans content by removing newline characters, extra spaces, and specific UI phrases.
    """
    content = content.replace('\\n', ' ').replace('\n', ' ')
    content = content.replace('\\t', ' ').replace('\t', ' ')
    content = content.replace('\\r', ' ').replace('\r', ' ')
    content = content.replace('\"', '')
    content = content.replace('Copy to clipboard', '').replace('Error', '').replace('Copied', '')
    content = ' '.join(content.split())
    return content

def retrieve_and_prioritize_documents(question_embedding, index, metadata, top_k_initial=750, top_k_final=5):
    """
    Retrieves and prioritizes documents based on question embedding similarity.
    Now directly uses FAISS distances for scoring as embeddings are no longer in metadata.
    """
    D, I = index.search(question_embedding.reshape(1, -1), top_k_initial)
    
    retrieved_docs_raw = []
    for i, dist in zip(I[0], D[0]):
        doc = metadata[int(i)] 
        doc["score"] = -dist # <-- CORRECTED: Use negative FAISS distance as score
        retrieved_docs_raw.append(doc)

    retrieved_docs = []
    for doc in retrieved_docs_raw:
        content = doc.get("content", "")
        if not content or content.strip().endswith("?") or len(content.strip()) < 10:
            continue
        retrieved_docs.append(doc)

    retrieved_docs.sort(key=lambda x: x["score"], reverse=True)
    
    final_relevant_docs = []
    for doc in retrieved_docs:
        if len(final_relevant_docs) < top_k_final:
            doc["url"] = format_discourse_url(doc)
            final_relevant_docs.append(doc)
        else:
            break

    return final_relevant_docs


def build_prompt(question, relevant_docs, image_url=None):
    """
    Constructs the prompt messages for the LLM.
    Further refined to guide the LLM on making recommendations for choices,
    especially concerning specific project requirements.
    """
    messages = []
    
    messages.append({
        "role": "system",
        "content": (
            "You are a highly capable and thorough Virtual Teaching Assistant for the IIT Madras Online Degree in Data Science. "
            "Your primary goal is to provide comprehensive and detailed answers based *only* on the provided context in metadata.json embeddings, ensuring factual accuracy from the sources. "
            "**If the user's question presents a choice (e.g., 'should I use X or Y?') or asks for a recommendation, carefully analyze the provided context to identify the supported or required option for *this project's specific setup or course instructions*. Do not provide answers based on general GPT answers. Provide answers only from the embeddings** "
            "**Explicitly state which option is appropriate based on the context, and briefly explain why other options may not be suitable or are not explicitly mentioned as supported within the project's framework.** "
            "If the answer cannot be found in the context, state that clearly and briefly mention that the information is not available in the provided sources. "
            "Format your answer as a JSON object with 'answer' (the generated answer) "
            "and 'links' (an array of objects, each with 'url' and 'text'). "
            "For each distinct piece of information used in your answer, cite the corresponding source URL and a short, descriptive text snippet (max 150 characters, human-readable, no code/escapes) directly from that source that *specifically supports* that part of your answer. "
            "**You MUST provide exactly 2 highly relevant and distinct links based on the context.** If the provided context does not contain enough truly distinct pieces of information to cite two unique links, you must still provide 2 links by citing the most relevant available sources, even if some have similar content. Prioritize sources that directly support your answer. "
            "**When multiple sources contain similar information, prioritize citing the source that appears earlier in the provided context (e.g., Source 1 over Source 2), as these are ordered by relevance.**"
            "Ensure the 'text' in links is a *human-readable* snippet directly from the source, not programmatic representations like '\\n' or escape characters. "
            "Do not include any preambles or explanations outside the JSON. Aim for a direct answer."
        )
    })

    user_content = []
    
    user_content.append({"type": "text", "text": f"Question: {question}"})
    
    if image_url:
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})

    context_str = ""
    for i, doc in enumerate(relevant_docs):
        cleaned_doc_content = clean_content_for_prompt(doc.get("content", ""))
        context_str += f"\n--- Source {i+1} ---\n"
        context_str += f"URL: {doc.get('url', 'N/A')}\n"
        context_str += f"Content: {cleaned_doc_content}\n"
    
    if context_str:
        user_content.append({"type": "text", "text": f"\nContext:\n{context_str}"})
    else:
        user_content.append({"type": "text", "text": "\nNo specific context documents were found relevant to your question, but I will try to answer based on general knowledge if possible, otherwise I will state I cannot answer."})

    user_content.append({"type": "text", "text": (
        'Now, provide your answer strictly in the following JSON format: '
        '{"answer": "...", "links": [{"url": "...", "text": "..."}, {"url": "...", "text": "..."}]}'
    )})

    messages.append({"role": "user", "content": user_content})
    
    return messages

def query_chat_completion(messages):
    """
    Sends messages to the chat completion endpoint and retrieves the raw response.
    """
    try:
        resp = requests.post(CHAT_ENDPOINT, json={"model": CHAT_MODEL, "messages": messages, "temperature": 0.3, "response_format": {"type": "json_object"}}, headers=HEADERS, timeout=120)
        resp.raise_for_status()
        raw_content = resp.json()["choices"][0]["message"]["content"]
        
        return raw_content
    except requests.exceptions.RequestException as e:
        print(f"Error querying chat completion from {CHAT_ENDPOINT}: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}", file=sys.stderr)
            print(f"Response body: {e.response.text}", file=sys.stderr)
        raise

def clean_json_response(raw_llm_response_text, relevant_docs):
    """
    Cleans and parses the JSON response from the LLM.
    Ensures exactly 2 unique and relevant links are included in the output if possible,
    by padding from relevant_docs if the LLM provides fewer.
    """
    try:
        text = raw_llm_response_text.strip()
        # Remove markdown code block delimiters
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        elif text.startswith("```"):
            text = text[len("```"):].strip()
        if text.endswith("```"): # Safely remove if it exists
            text = text[:-len("```")].strip()
        
        data = json.loads(text)
        
        if not isinstance(data.get("answer"), str):
            data["answer"] = str(data.get("answer", ""))
        
        if not isinstance(data.get("links"), list):
            data["links"] = []
        
        final_links = []
        seen_urls = set()

        # 1. Collect links explicitly cited by the LLM (prioritize these)
        for link in data["links"]:
            if isinstance(link, dict) and "url" in link and "text" in link:
                url = link["url"].strip()
                text_snippet = clean_content_for_prompt(link["text"])
                
                if len(text_snippet) > 150:
                    text_snippet = text_snippet[:150].rsplit(' ', 1)[0] + '...'

                if url and text_snippet and url not in seen_urls and len(final_links) < 2:
                    final_links.append({"url": url, "text": text_snippet})
                    seen_urls.add(url)
        
        # 2. If fewer than 2 links, pad from the top relevant_docs
        for doc in relevant_docs:
            if len(final_links) >= 2:
                break

            doc_url = doc.get("url", "").strip()
            doc_content_snippet = clean_content_for_prompt(doc.get("content", ""))
            
            snippet_for_padding = doc_content_snippet[:150].rsplit(' ', 1)[0] + '...' if len(doc_content_snippet) > 150 else doc_content_snippet

            if doc_url and snippet_for_padding and doc_url not in seen_urls:
                final_links.append({"url": doc_url, "text": snippet_for_padding})
                seen_urls.add(doc_url)
        
        data["links"] = final_links[:2]
        
        return data
    except json.JSONDecodeError as e:
        print(f"JSON decoding error in clean_json_response: {e}", file=sys.stderr)
        print(f"Raw response that caused error: {text}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while cleaning JSON: {e}", file=sys.stderr)
        print(f"Raw response that caused error: {text}", file=sys.stderr)
        return None

def encode_image_to_base64_data_uri(image_path):
    """
    Encodes an image file to a base64 data URI for embedding in multimodal prompts.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{encoded_string}"