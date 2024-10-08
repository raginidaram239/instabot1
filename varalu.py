import os
import faiss
import pickle
import numpy as np
import openai
import time
import streamlit as st
from indic_transliteration import sanscript

from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

# Paths to the FAISS index and chunk files
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_FILE_PATH = "chunks.pkl"
EMBEDDINGS_FILE_PATH = "embeddings.pkl"
TEXT_FILE_PATH = "fair (1).txt"

# OpenAI API details
openai.api_type = "azure"
openai.api_base = os.getenv('OPENAI_API_BASE', "https://aj-open-ai.openai.azure.com/")
openai.api_version = "2023-12-01-preview"
openai.api_key = os.getenv('OPENAI_API_KEY', "7ec51e85779e41bca5bc29b9e533ee47")

# Function to read text from a local file
def read_text_from_local(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return ""

# Function to chunk text by paragraphs
def chunk_text_by_paragraphs(text, max_chunk_size=1000):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to generate embeddings in batches
def generate_embeddings_in_batches(chunks, batch_size=10):
    embeddings = []
    max_tokens = 8192  # Adjust based on OpenAI model's token limit
    for i in range(0, len(chunks), batch_size):
        batch_chunks = [chunk for chunk in chunks[i:i + batch_size] if chunk.strip() and len(chunk) <= max_tokens]

        if not batch_chunks:
            print(f"Skipping empty or invalid chunks in batch {i // batch_size + 1}.")
            continue

        try:
            response = openai.Embedding.create(
                deployment_id="text-embedding-ada-002",
                input=batch_chunks
            )
            batch_embeddings = [embedding['embedding'] for embedding in response['data']]
            embeddings.extend(batch_embeddings)
            print(f"Generated embeddings for batch {i // batch_size + 1}.")
            time.sleep(2)
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error generating embeddings: {e}")
            time.sleep(10)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            time.sleep(10)

    return embeddings

# Function to generate and save FAISS index
def generate_and_save_faiss_index(chunks, embeddings):
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Total embeddings generated: {len(embeddings)}")

    try:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

    try:
        with open(CHUNKS_FILE_PATH, "wb") as f:
            pickle.dump(chunks, f)
        print(f"Chunks saved to {CHUNKS_FILE_PATH}")
    except Exception as e:
        print(f"Error saving chunks: {e}")

    try:
        with open(EMBEDDINGS_FILE_PATH, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {EMBEDDINGS_FILE_PATH}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

# Helper function to normalize embeddings
def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

# Load FAISS index
def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_PATH)

# Load chunks
def load_chunks():
    with open(CHUNKS_FILE_PATH, "rb") as f:
        return pickle.load(f)

# Load embeddings
def load_embeddings():
    with open(EMBEDDINGS_FILE_PATH, "rb") as f:
        return pickle.load(f)

# Embed query using OpenAI API
def embed_query(query):
    try:
        response = openai.Embedding.create(
            deployment_id="text-embedding-ada-002",
            input=[query]
        )
        embedding = response['data'][0]['embedding']
        return normalize_embedding(embedding)
    except Exception as e:
        st.error(f"Error generating embedding for query: {e}")
        return None

# Query the FAISS index
def query_faiss_index(index, query_embedding, k=5):
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
    return indices[0] if len(indices) > 0 else []

# Rerank chunks based on cosine similarity
def rerank_chunks(query_embedding, chunks, retrieved_indices, chunk_embeddings):
    similarities = []
    for i in retrieved_indices:
        chunk_embedding = chunk_embeddings[i]
        similarity = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
        similarities.append((chunks[i], similarity))
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Select top relevant chunks
def select_top_chunks(relevant_chunks, top_n=5):
    return " ".join([chunk for chunk, _ in relevant_chunks[:top_n]])

# Generate dynamic prompt
def get_dynamic_prompt(query, context):
    return f"Based on the following context, answer the query briefly: '{query}'\n\nContext:\n{context}\n\nPlease provide a short and to-the-point answer."

# Get GPT-35-Turbo response
def get_gpt_response(messages):
    try:
        response = openai.ChatCompletion.create(
            deployment_id="gpt-35-turbo",
            messages=messages,
            max_tokens=80
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error getting GPT response: {e}")
        return "Error generating response."

# Language detection and transliteration functions
def is_tamilish_input(query):
    tamilish_keywords = ['eppadi', 'irukkiRaai', 'vandhirukka', 'enakku']
    return any(keyword in query.lower() for keyword in tamilish_keywords)

def transliterate_to_tamil(query):
    return transliterate(query, sanscript.ITRANS, sanscript.TAMIL)

def is_telugu_input(query):
    telugu_keywords = ['prathi', 'sambandham', 'varthakudu', 'mithrudu']
    return any(keyword in query.lower() for keyword in telugu_keywords)

def transliterate_to_telugu(query):
    return transliterate(query, sanscript.ITRANS, sanscript.TELUGU)

def is_sinhala_input(query):
    sinhala_keywords = ['kramaya', 'pavathi', 'saha', 'bala']
    return any(keyword in query.lower() for keyword in sinhala_keywords)

def transliterate_to_sinhala(query):
    return transliterate(query, sanscript.ITRANS, sanscript.SINHALA)

# Main function for processing text, generating index, and serving Streamlit app
def main():
    # Streamlit UI
    st.markdown("""<style>
    .title { text-align: center; color: #003366; font-size: 36px; margin-bottom: 20px; }
    .content-box { padding: 15px; background-color: #F5F5F5; border-radius: 8px; margin-bottom: 20px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); }
    .button { background-color: #003366; color: #FFFFFF; }
    .button:hover { background-color: #002244; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>Unilever Multi Product AI Agent</h1>", unsafe_allow_html=True)
    
    # Logo
    st.image("path/to/your/logo.png", width=200)  # Update with the path to your logo

    # Text processing and FAISS index generation
    text = read_text_from_local(TEXT_FILE_PATH)
    if text:
        chunks = chunk_text_by_paragraphs(text)
        print(f"Total chunks created: {len(chunks)}")
        embeddings = generate_embeddings_in_batches(chunks)
        if embeddings:
            generate_and_save_faiss_index(chunks, embeddings)

    # Initialize conversation memory
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Query input
    query = st.text_input("Enter your FAQ:", placeholder="Type your query here...", help="Input your question or query here.", max_chars=150)
    submit_query = st.button("Submit Query", key="submit_query", help="Click to submit your query.", use_container_width=True)

    if submit_query and query:
        index = load_faiss_index()
        chunks = load_chunks()
        embeddings = load_embeddings()

        # Detect language and transliterate if necessary
        if is_tamilish_input(query):
            query = transliterate_to_tamil(query)
        elif is_telugu_input(query):
            query = transliterate_to_telugu(query)
        elif is_sinhala_input(query):
            query = transliterate_to_sinhala(query)

        query_embedding = embed_query(query)
        if query_embedding is not None:
            retrieved_indices = query_faiss_index(index, query_embedding)
            relevant_chunks = rerank_chunks(query_embedding, chunks, retrieved_indices, embeddings)
            context = select_top_chunks(relevant_chunks)

            messages = [{"role": "user", "content": get_dynamic_prompt(query, context)}]
            answer = get_gpt_response(messages)

            # Add user query and AI response to conversation history
            st.session_state.conversation.append({"user": query, "ai": answer})

            # Display AI response
            st.markdown("### AI Response:")
            st.write(answer)

            # Display conversation history
            if st.session_state.conversation:
                st.markdown("### Conversation History:")
                for entry in st.session_state.conversation:
                    st.markdown(f"**User:** {entry['user']}")
                    st.markdown(f"**AI:** {entry['ai']}")

if __name__ == "__main__":
    main()
