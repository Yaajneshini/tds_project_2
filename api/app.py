from flask import Flask, request, jsonify
import os
import sys
import json

# Add current directory to sys.path so rag.py can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag import (
    load_index_and_metadata, get_embedding, retrieve_and_prioritize_documents,
    build_prompt, query_chat_completion, clean_json_response,
    encode_image_to_base64_data_uri, format_discourse_url, clean_content_for_prompt
)

app = Flask(__name__)

@app.route('/') # Add this route
def root_check():
    return 'Root OK', 200

@app.route('/healthz')
def health_check():
    return 'OK', 200

@app.route('/api/', methods=['POST'])
def answer_question_api():
    """
    POST endpoint: expects {"question": "...", "image": optional base64 or URL}
    Returns: {"answer": "...", "links": [{url, text}, ...]}
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    question = data.get("question")
    image_base64_data = data.get("image")

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    try:
        rag_index, rag_metadata = load_index_and_metadata()
    except Exception as e:
        print(f"RAG load error: {e}", file=sys.stderr)
        return jsonify({"answer": "Server is initializing or failed to load knowledge base. Try again later.", "links": []}), 503

    image_url_for_llm = None
    if image_base64_data:
        if image_base64_data.startswith("http://") or image_base64_data.startswith("https://"):
            image_url_for_llm = image_base64_data
        else:
            mime_type = "application/octet-stream"
            if image_base64_data.startswith("/9j/"):
                mime_type = "image/jpeg"
            elif image_base64_data.startswith("iVBORw0KGgo"):
                mime_type = "image/png"
            elif image_base64_data.startswith("UklGR"):
                mime_type = "image/webp"
            image_url_for_llm = f"data:{mime_type};base64,{image_base64_data}"

    try:
        q_embed = get_embedding(question)
    except Exception as e:
        print(f"Embedding error: {e}", file=sys.stderr)
        return jsonify({"answer": f"Error generating embedding: {e}", "links": []}), 500

    try:
        relevant_docs = retrieve_and_prioritize_documents(
            q_embed, rag_index, rag_metadata, top_k_initial=750, top_k_final=5
        )
    except Exception as e:
        print(f"Retrieval error: {e}", file=sys.stderr)
        return jsonify({"answer": f"Error retrieving documents: {e}", "links": []}), 500

    if not relevant_docs:
        return jsonify({"answer": "Sorry, no relevant content found in the knowledge base.", "links": []}), 200

    try:
        messages = build_prompt(question, relevant_docs, image_url=image_url_for_llm)
        raw_response = query_chat_completion(messages)
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return jsonify({"answer": f"Error querying LLM: {e}", "links": []}), 500

    answer_data = clean_json_response(raw_response, relevant_docs)

    if not answer_data or not answer_data.get("answer"):
        print("LLM returned malformed answer.", file=sys.stderr)
        fallback_text = "Couldn't generate a proper answer. Here's relevant info:\n"
        fallback_links = []
        for doc in relevant_docs[:2]:
            snippet = clean_content_for_prompt(doc.get("content", ""))[:150].rsplit(' ', 1)[0] + '...'
            fallback_links.append({"url": doc["url"], "text": snippet})
        if fallback_links:
            for link in fallback_links:
                fallback_text += f"- \"{link['text']}\" (Source: {link['url']})\n"
        else:
            fallback_text += "No fallback links available."
        return jsonify({"answer": fallback_text.strip(), "links": fallback_links}), 200

    answer_data["answer"] = answer_data["answer"].replace("\n", " ").strip()
    return jsonify(answer_data), 200

# ✅ Do NOT preload RAG assets here — lazy load inside handler
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
