import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Scrape Website Content
def scrape_website(url):
    """
    Extract meaningful paragraphs from a website, excluding short or irrelevant content.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [
            p.get_text().strip() 
            for p in soup.find_all('p') 
            if len(p.get_text().strip()) > 30
        ]
        return paragraphs
    except Exception as e:
        print(f"Error scraping website: {e}")
        return []

# Step 2: Convert Text to Embeddings
def create_embeddings(chunks, model):
    """
    Convert chunks of text into vector embeddings.
    """
    return model.encode(chunks, convert_to_tensor=True).cpu().numpy()

# Step 3: Build FAISS Index
def build_faiss_index(embeddings):
    """
    Create a FAISS index for similarity search.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)
    return index

# Step 4: Search and Rank Results
def search_query(query, model, index, chunks):
    """
    Search for the most relevant chunks based on query and rank results by similarity.
    """
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k=10)  # Fetch more results (e.g., top 10)
    ranked_results = sorted(
        [(chunks[i], distances[0][rank]) for rank, i in enumerate(indices[0])],
        key=lambda x: x[1]
    )
    return [result[0] for result in ranked_results[:3]]

# Main Function
def main():
    url = "https://www.stanford.edu/"  # Example website

    # Load Model
    print("Loading embedding model...")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Scrape Content
    print(f"Scraping website: {url}")
    chunks = scrape_website(url)
    if not chunks:
        print("No content retrieved from the website.")
        return
    print(f"Retrieved {len(chunks)} text chunks.")

    # Create Embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings(chunks, model)

    # Build FAISS Index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # User Queries
    while True:
        query = input("\nAsk your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        print("Searching for answers...")
        results = search_query(query, model, index, chunks)
        print("\nTop Results:")
        for i, result in enumerate(results):
            print(f"{i + 1}. {result}\n{'-' * 50}")

if __name__ == "__main__":
    main()
