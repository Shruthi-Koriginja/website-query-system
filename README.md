# website-query-system
A Retrieval-Augmented Generation (RAG) pipeline for interacting with websites, scraping data, and answering queries using FAISS and SentenceTransformers.

# Web RAG Pipeline

## Overview
The Web Retrieval-Augmented Generation (RAG) Pipeline allows users to scrape website content, process it into embeddings, and retrieve relevant information based on user queries. It uses AI-powered models for embedding generation and FAISS for similarity-based search, providing accurate results for natural language questions.

---

## Features
1. **Website Scraping**: Extracts meaningful content from the specified website.
2. **Embeddings**: Converts text into vector embeddings using `SentenceTransformers`.
3. **FAISS Index**: Efficiently stores embeddings for similarity-based retrieval.
4. **Query Ranking**: Matches user queries to the most relevant content and ranks results.
5. **Improved Filtering**: Excludes irrelevant or low-quality content during scraping.

---

## Requirements
To run this project, install the following Python libraries:
- `requests`: For making HTTP requests to scrape website content.
- `beautifulsoup4`: For parsing HTML and extracting textual data.
- `sentence-transformers`: For generating embeddings from text.
- `faiss-cpu`: For performing fast similarity searches on embeddings.

You can install these libraries using the provided `requirements.txt` file.

---

## Installation and Setup

### Prerequisites
- Python 3.6 or higher.
- Internet connection for downloading dependencies and pre-trained models.

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/web_rag_pipeline.git
   cd web_rag_pipeline


## How to Run 
1]Clone this repository:
git clone https://github.com/<Shruthi-Koriginja>/web_rag_pipeline.git
cd web_rag_pipeline

2]Install dependencies:
pip install -r requirements.txt

3]Run the main script:
python web_rag_pipeline.py

## How to Run (in detail)

1]Run the main Python script:
  python web_RAG_Pipeline.py
2]The script will prompt for a website URL to scrape.
3]Enter your query (e.g., "What is Stanford known for?") to retrieve relevant answers.
4]To exit, type exit

## Example
Input:
Ask your question (or type 'exit' to quit): What is Stanford known for?
## Output:
Top Results:
1. Stanford was founded almost 150 years ago on a bedrock of societal purpose. Our mission is to contribute to the world by educating students for lives of leadership and contribution with integrity; advancing fundamental knowledge and cultivating creativity; leading in pioneering research for effective clinical therapies; and accelerating solutions and amplifying their impact.
--------------------------------------------------
2. Another relevant paragraph from the website...
--------------------------------------------------
3. Yet another relevant paragraph...
--------------------------------------------------






