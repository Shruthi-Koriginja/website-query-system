# Required Libraries
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Crawl and Scrape Website Content
def scrape_website(url):
    """
    Fetch and parse text content from a website.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch the website")
        return []
    
    # Parse content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    return paragraphs

# Step 2: Create Embeddings for Content
def create_embeddings(text_chunks, model):
    """
    Convert text into vector embeddings using a pre-trained model.
    """
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    return np.array(embeddings.cpu())

# Step 3: Build FAISS Index for Search
def build_faiss_index(embeddings):
    """
    Create a FAISS index for fast similarity searches.
    """
    dimension = embeddings.shape[1]  # Size of each embedding vector
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(embeddings)
    return index

# Step 4: Search and Retrieve Relevant Chunks
def search_query(query, model, index, text_chunks):
    """
    Search for the most relevant text chunks based on the query.
    """
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k=3)  # Top 3 results
    results = [text_chunks[i] for i in indices[0]]
    return results

# Step 5: Main Function for End-to-End Pipeline
def main():
    url = "https://www.stanford.edu/"  # Example website
    
    # Load pre-trained embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Scrape content
    print(f"Scraping website: {url}")
    text_chunks = scrape_website(url)
    if not text_chunks:
        print("No content retrieved from the website.")
        return
    print(f"Retrieved {len(text_chunks)} text chunks.")

    # Create embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings(text_chunks, model)

    # Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # User query input
    while True:
        query = input("\nAsk your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        # Search for relevant content
        print("Searching for answers...")
        results = search_query(query, model, index, text_chunks)
        
        # Display results
        print("\nTop Results:")
        for idx, res in enumerate(results):
            print(f"{idx+1}. {res}\n")

if __name__ == "__main__":
    main()
#C:\Users\Dell\OneDrive\Desktop
