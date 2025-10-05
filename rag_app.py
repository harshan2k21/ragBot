import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, pipeline
import torch
from flask import Flask, request, jsonify, render_template_string

# --- Configuration ---
# Use an environment variable for the app's root directory, fallback for local dev
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

DOCS_URL = "https://requests.readthedocs.io/en/latest/"
INDEX_FILE = os.path.join(APP_ROOT, "docs.index")
DOC_CONTENT_FILE = os.path.join(APP_ROOT, "docs_content.txt")

# --- Global Variables ---
# These will be initialized by initialize_rag_pipeline()
retriever = None
qa_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- HTML & CSS Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Q&A Bot</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="bg-white p-8 rounded-lg shadow-lg w-full max-w-2xl">
        <h1 class="text-2xl font-bold mb-4 text-center text-gray-800">RAG-Powered Q&A Bot</h1>
        <p class="text-gray-600 mb-6 text-center">Ask a question about the Python 'requests' library.</p>
        <div class="mb-4">
            <input type="text" id="questionInput" class="w-full px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="e.g., How to add headers to a request?">
        </div>
        <div class="text-center mb-6">
            <button id="askButton" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-6 rounded-md transition duration-300">Ask</button>
        </div>
        <div id="response" class="p-4 bg-gray-50 rounded-md border min-h-[100px]">
             <p class="text-gray-500">Your answer will appear here...</p>
        </div>
        <div id="loader" class="hidden flex justify-center mt-4"><div class="loader"></div></div>
    </div>

    <script>
        const askButton = document.getElementById('askButton');
        const questionInput = document.getElementById('questionInput');
        const responseDiv = document.getElementById('response');
        const loader = document.getElementById('loader');

        askButton.addEventListener('click', askQuestion);
        questionInput.addEventListener('keyup', (event) => {
            if (event.key === 'Enter') {
                askQuestion();
            }
        });

        async function askQuestion() {
            const question = questionInput.value;
            if (!question) {
                responseDiv.innerHTML = '<p class="text-red-500">Please enter a question.</p>';
                return;
            }

            loader.classList.remove('hidden');
            responseDiv.innerHTML = '<p class="text-gray-500">Thinking...</p>';
            askButton.disabled = true;

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                responseDiv.innerHTML = `<p class="text-gray-800">${data.answer.replace(/\\n/g, '<br>')}</p>`;

            } catch (error) {
                console.error('Error:', error);
                responseDiv.innerHTML = '<p class="text-red-500">Sorry, something went wrong. Please check the console for details.</p>';
            } finally {
                loader.classList.add('hidden');
                askButton.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

# --- Core RAG Logic ---

def initialize_rag_pipeline():
    """Initializes the entire RAG pipeline."""
    print("Initializing RAG pipeline...")
    global qa_pipeline, retriever
    
    # Check if index files exist. If not, create them.
    if not os.path.exists(INDEX_FILE) or not os.path.exists(DOC_CONTENT_FILE):
        print(f"'{INDEX_FILE}' or '{DOC_CONTENT_FILE}' not found. Starting build process...")
        
        # 1. Scrape Website
        print(f"Scraping content from {DOCS_URL}...")
        try:
            response = requests.get(DOCS_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(separator='\n', strip=True)
            print("Scraping successful.")
        except requests.RequestException as e:
            print(f"Error scraping website: {e}")
            return

        # 2. Chunk Documents
        documents = [para.strip() for para in text_content.split('\n') if len(para.strip()) > 50]
        print(f"Created {len(documents)} document chunks.")

        # 3. Create Embeddings and Index
        print("Loading sentence transformer model for embeddings...")
        embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
        embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        print("Generating embeddings for documents... (This may take a while)")
        embeddings = []
        for doc in documents:
            inputs = embedding_tokenizer(doc, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                output = embedding_model(**inputs)
            embeddings.append(output.last_hidden_state.mean(dim=1).cpu().numpy())
        
        doc_embeddings = np.vstack(embeddings)
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings)
        print("FAISS index created successfully.")

        # 4. Save the index and documents
        faiss.write_index(index, INDEX_FILE)
        with open(DOC_CONTENT_FILE, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(f"{doc}\n")
        print(f"Index saved to '{INDEX_FILE}', content saved to '{DOC_CONTENT_FILE}'.")

    # 5. Load the models and the saved index/documents for the application
    print("Loading models and pre-built index...")
    embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    index = faiss.read_index(INDEX_FILE)
    documents = [line.strip() for line in open(DOC_CONTENT_FILE, 'r', encoding='utf-8')]
    
    def retrieve_documents(query, k=5):
        """Retrieve top-k documents from the FAISS index."""
        inputs = embedding_tokenizer(query, return_tensors='pt').to(device)
        with torch.no_grad():
            output = embedding_model(**inputs)
        query_embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()
        
        _, indices = index.search(query_embedding, k)
        return [documents[i] for i in indices[0]]

    retriever = retrieve_documents
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad", device=0 if device=="cuda" else -1)
    
    print("RAG pipeline is ready!")


# --- Flask Web Server ---
app = Flask(__name__)

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    """Handles the question answering logic."""
    if not retriever or not qa_pipeline:
        return jsonify({"error": "RAG pipeline not initialized"}), 500
        
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 1. Retrieve relevant documents
    retrieved_docs = retriever(question)
    context = " ".join(retrieved_docs)

    # 2. Use QA model to find the answer in the context
    result = qa_pipeline(question=question, context=context)
    
    return jsonify({"answer": result['answer']})

if __name__ == '__main__':
    initialize_rag_pipeline()
    # Note: Using debug=False for production-like local testing.
    # For actual debugging, you might want to set it to True.
    app.run(host='0.0.0.0', port=5000, debug=False)