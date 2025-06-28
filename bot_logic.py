import textwrap
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, chunk_size)

def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_file(uploaded_file, file_type):
    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    chunks = chunk_text(text)
    vectorizer = TfidfVectorizer().fit(chunks)
    chunk_vectors = vectorizer.transform(chunks)

    return chunks, vectorizer, chunk_vectors

def get_relevant_chunks(query, vectorizer, chunk_vectors, chunks):
    vec = vectorizer.transform([query])
    similarities = cosine_similarity(vec, chunk_vectors).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    return "\n\n".join([chunks[i] for i in top_indices])

def build_prompt(query, history, vectorizer, chunk_vectors, chunks):
    chat_history = "\n".join([f"User: {u}\nBot: {b}" for u, b in history[-3:]])
    context = get_relevant_chunks(query, vectorizer, chunk_vectors, chunks)
    prompt = f"""You are a helpful assistant, that answers based on the context and document provided. Use the provided context and chat history to answer the user's question.

Context:
{context}

Chat History:
{chat_history}

User: {query}
Bot:"""
    return prompt
