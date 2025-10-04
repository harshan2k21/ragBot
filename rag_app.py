import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from flask import Flask, request, jsonify, render_template_string

# --- 1. Configuration ---
# You can change this URL to any documentation site you want to scrape.
# Let's use the documentation for `requests`, a popular Python library.
DOCS_URL = "https://requests.readthedocs.io/en/latest/"
INDEX_FILE = "docs.index"
DOC_CONTENT_FILE = "docs_content.txt"

# --- 2. Flask App Initialization ---
app = Flask(__name__)

# --- 3. Global Variables for AI Models and Data ---
# We load these globally to avoid reloading them on every request, which would be very slow.
vectorizer = None
qa_pipeline = None
index = None
documents = []

# --- 4. Core RAG Pipeline Functions ---

def scrape_documentation(url):
    """
    Scrapes text content from the given URL.
    This is a simple scraper and might need to be adapted for different website structures.
    """
    print(f"Scraping documentation from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the main content area of the documentation
        # Note: The class name 'document' is common in Sphinx-based docs like requests'.
        content = soup.find('div', role='main') or soup.find('div', class_='document')
        if not content:
            # Fallback to body if specific content divs aren't found
            content = soup.body

        text = content.get_text(separator='\n', strip=True)
        print("Scraping complete.")
        return text
    except requests.RequestException as e:
        print(f"Error scraping documentation: {e}")
        return ""

def preprocess_and_chunk(text):
    """
    Splits the scraped text into smaller, manageable chunks (e.g., paragraphs).
    """
    print("Preprocessing and chunking text...")
    # Split by double newlines (common paragraph separator) and filter out empty lines.
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    print(f"Created {len(chunks)} chunks.")
    return chunks

def build_and_save_vector_index(chunks, vectorizer_model):
    """
    Creates vector embeddings for each text chunk and builds a FAISS index for fast searching.
    """
    print("Creating vector embeddings for chunks...")
    embeddings = vectorizer_model.encode(chunks, show_progress_bar=True)
    
    print("Building FAISS index...")
    d = embeddings.shape[1]  # Get the dimension of the vectors
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(embeddings)
    
    print(f"Saving FAISS index to {INDEX_FILE}...")
    faiss.write_index(faiss_index, INDEX_FILE)

    # Also save the original text chunks for context retrieval
    with open(DOC_CONTENT_FILE, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n") # Use a separator
    
    print("Index and content saved.")
    return faiss_index

def retrieve_context(query, vectorizer_model, faiss_index, docs, k=5):
    """
    Finds the most relevant document chunks for a given query using the FAISS index.
    """
    print(f"Retrieving context for query: '{query}'")
    query_vector = vectorizer_model.encode([query])
    distances, indices = faiss_index.search(query_vector, k)
    
    # Retrieve the actual text chunks based on the indices
    retrieved_docs = [docs[i] for i in indices[0]]
    context = " ".join(retrieved_docs)
    print("Context retrieved.")
    return context

# --- 5. Initial Setup Function ---
def initialize_rag_pipeline():
    """
    This function orchestrates the entire setup process.
    It checks if an index exists and either loads it or builds a new one.
    """
    global vectorizer, qa_pipeline, index, documents

    print("Initializing RAG pipeline...")
    # Load the model for creating vector embeddings.
    # 'all-MiniLM-L6-v2' is a great balance of speed and quality.
    vectorizer = SentenceTransformer('all-MiniLM-L6-v2')

    # Load a pre-trained model for Question-Answering.
    # 'distilbert-base-cased-distilled-squad' is a smaller, faster model.
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

    if os.path.exists(INDEX_FILE) and os.path.exists(DOC_CONTENT_FILE):
        print("Loading existing index and documents...")
        index = faiss.read_index(INDEX_FILE)
        with open(DOC_CONTENT_FILE, 'r', encoding='utf-8') as f:
            documents = f.read().split("\n---\n")
    else:
        print("No index found. Building a new one...")
        raw_text = scrape_documentation(DOCS_URL)
        documents = preprocess_and_chunk(raw_text)
        index = build_and_save_vector_index(documents, vectorizer)
    
    print("RAG Pipeline is ready!")

# --- 6. Flask Web Routes ---

# The HTML template is defined directly in the Python string for simplicity.
# It uses Tailwind CSS for modern styling.
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
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container mx-auto p-4 md:p-8 max-w-3xl">
        <div class="bg-white rounded-xl shadow-lg p-6 md:p-8">
            <h1 class="text-3xl md:text-4xl font-bold text-center text-gray-900 mb-2">Documentation Q&A Bot</h1>
            <p class="text-center text-gray-500 mb-6">Ask a question about the 'requests' Python library.</p>
            
            <form id="qa-form" class="flex flex-col sm:flex-row gap-3">
                <input type="text" id="question" class="flex-grow p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition" placeholder="e.g., How do I send a POST request?">
                <button type="submit" class="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition duration-300 flex items-center justify-center">
                    <span id="button-text">Ask</span>
                    <div id="loader" class="loader hidden ml-3"></div>
                </button>
            </form>

            <div id="answer-container" class="mt-8 p-6 bg-gray-50 rounded-lg border border-gray-200 hidden">
                <h2 class="text-xl font-semibold mb-2 text-gray-800">Answer:</h2>
                <p id="answer" class="text-gray-700 whitespace-pre-wrap"></p>
            </div>
        </div>
        <footer class="text-center mt-6 text-sm text-gray-400">
            <p>Powered by RAG & Flask</p>
        </footer>
    </div>

    <script>
        const form = document.getElementById('qa-form');
        const questionInput = document.getElementById('question');
        const answerContainer = document.getElementById('answer-container');
        const answerEl = document.getElementById('answer');
        const buttonText = document.getElementById('button-text');
        const loader = document.getElementById('loader');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = questionInput.value;
            if (!question.trim()) return;

            // Show loader and disable form
            buttonText.classList.add('hidden');
            loader.classList.remove('hidden');
            form.querySelector('button').disabled = true;
            answerContainer.classList.add('hidden');

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();
                
                answerEl.textContent = data.answer || "Sorry, I couldn't find an answer in the documentation.";
                answerContainer.classList.remove('hidden');

            } catch (error) {
                console.error('Error:', error);
                answerEl.textContent = 'An error occurred while fetching the answer. Please try again.';
                answerContainer.classList.remove('hidden');
            } finally {
                // Hide loader and re-enable form
                buttonText.classList.remove('hidden');
                loader.classList.add('hidden');
                form.querySelector('button').disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handles the question from the frontend, runs the RAG pipeline, and returns the answer.
    """
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    # 1. Retrieve relevant context from the vector database
    context = retrieve_context(question, vectorizer, index, documents)

    # 2. Use the QA model to generate an answer based on the context
    result = qa_pipeline(question=question, context=context)
    
    return jsonify({'answer': result['answer']})


# --- 7. Main Execution Block ---
if __name__ == '__main__':
    # Initialize the models and data pipeline before starting the web server
    initialize_rag_pipeline()
    # Run the Flask app
    app.run(debug=True)