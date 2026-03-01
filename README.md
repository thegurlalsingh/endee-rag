# Medical Encyclopedia RAG with Endee Vector Database

- Extracts and chunks text from a PDF (Gale Encyclopedia of Medicine)
- Generates high-quality embeddings using a BAAI/bge-m3 **on Google Colab** 
- Downloads embeddings + metadata
- Upserts them into a **local Endee** vector database
- Enables semantic search queries

## Project Overview and Problem Statement

**Goal**: Build a local, private RAG system to answer questions about medical topics using content from a large reference PDF (Gale Encyclopedia of Medicine).

**Problem**:
- PDFs (especially encyclopedias) are large, unstructured, and hard to search semantically.
- Running large embedding models locally is slow / memory-intensive.
- Most cloud vector DBs require internet / cost money for experimentation.
- Need fast, accurate retrieval for real-time question answering.

**Solution**:
- Offload heavy embedding computation to free Colab GPU.
- Use Endee locally for storage & search.
- Achieve low-latency semantic search (sub-10 ms queries possible with Endee) on medical knowledge without sending data to the cloud.

## System Design & Technical Approach

PDF Document
(Gale Encyclopedia of Medicine or similar medical reference)
|
manual upload
|
Google Colab 
├── PDF text extraction → PyPDF2
├── Semantic chunking with overlap → LangChain RecursiveCharacterTextSplitter
├── Dense embedding generation → Sentence-Transformers (BAAI/bge-m3)
└── Export → embeddings.npy + chunks_metadata.json
│
manual file transfer
|
Local Machine 
├── Endee vector database server
└── Python client → create index → batch ingestion
│
Query Time 
├── User question → encode with same embedding model 
├── Endee ANN search → top-k similar chunks

## How Endee Is Used

Endee serves as the **persistent, high-performance vector store** and **real-time retrieval engine**.

## Setup & Execution Instructions

### 1. Local Endee Installation (one-time)

```bash
git clone https://github.com/endee-io/endee.git
cd endee
```

### Build & install 
```bash
chmod +x install.sh
./install.sh --release --neon
```

### Start server
```bash
chmod +x run.sh
./run.sh
```

## 2.Generate Embeddings via Google Colab/Local Machine
Run create_embeddings.ipynb and store all the embeddings

## 3.Feed all the embeddings in main.py and ingest them on Endee Vector Database

## 4.Enter your query in test.py and run

### NOTE : All the packages must be installed properly
