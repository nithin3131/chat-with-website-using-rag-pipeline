import requests
from bs4 import BeautifulSoup
import uuid
import faiss
import numpy as np
import openai
from flask import Flask, request, jsonify

# Check if the sentence_transformers library is available
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except ModuleNotFoundError:
    embedding_model = None
    print("Error: The 'sentence_transformers' library is not installed. Please install it using 'pip install sentence-transformers'.")

# Initialize FAISS index
dimension = 384  # Embedding size of MiniLM
index = faiss.IndexFlatL2(dimension)

# Metadata store
meta_store = []

# Set your OpenAI API key
openai.api_key = "your-api-key"

# Function to scrape and chunk content
def scrape_and_chunk(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract textual content
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    content = " ".join(paragraphs)

    # Split content into chunks (e.g., 300 words per chunk)
    chunk_size = 300
    words = content.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Generate embeddings and metadata if the model is available
    if embedding_model:
        embeddings = [embedding_model.encode(chunk) for chunk in chunks]
    else:
        embeddings = []

    metadata = [{"id": str(uuid.uuid4()), "url": url, "chunk": chunk} for chunk in chunks]

    return embeddings, metadata

# Function to add data to the FAISS index
def add_to_index(embeddings, metadata, index, meta_store):
    if embeddings:
        vectors = np.array(embeddings).astype('float32')
        index.add(vectors)

        # Store metadata separately
        meta_store.extend(metadata)

# Function to perform similarity search
def search(query, index, meta_store, top_k=5):
    if not embedding_model:
        print("Error: Embedding model is not available. Cannot perform search.")
        return []

    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve metadata for the top results
    results = [meta_store[i] for i in indices[0]]
    return results

# Function to generate response using OpenAI GPT
def generate_response(query, retrieved_chunks):
    context = "\n".join([f"- {chunk['chunk']}" for chunk in retrieved_chunks])
    prompt = f"""
    You are an intelligent assistant. Answer the following question based on the provided context:

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip()

# Flask app setup
app = Flask(_name_)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data['query']

    # Perform search and generate response
    results = search(query, index, meta_store)
    response = generate_response(query, results)

    return jsonify({"response": response})

if _name_ == '_main_':
    # Example: Scrape and index some websites before starting the app
    urls = ["https://www.uchicago.edu/", "https://www.stanford.edu/"]
    for url in urls:
        embeddings, metadata = scrape_and_chunk(url)
        add_to_index(embeddings, metadata, index, meta_store)

    app.run(debug=True)
