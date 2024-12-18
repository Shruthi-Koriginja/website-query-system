# website-query-system
A Retrieval-Augmented Generation (RAG) pipeline for interacting with websites, scraping data, and answering queries using FAISS and SentenceTransformers.

# Web RAG Pipeline

## Overview
The Web Retrieval-Augmented Generation (RAG) Pipeline allows users to query data from websites. It scrapes website content, processes it into embeddings, and retrieves answers based on user questions.

---

## Features
1. **Website Scraping**: Extracts text data from websites.
2. **Embeddings**: Converts content into embeddings using `SentenceTransformers`.
3. **FAISS Index**: Stores embeddings for fast similarity searches.
4. **Question Answering**: Matches user queries to relevant website content.

---

## Requirements
Install the following dependencies:
```bash
pip install -r requirements.txt

## How to Run
Clone this repository:
git clone https://github.com/<Shruthi-Koriginja>/web_rag_pipeline.git
cd web_rag_pipeline

## Install dependencies:
pip install -r requirements.txt

## Run the main script:
python web_rag_pipeline.py
