# AI Web Research Agent

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A professional AI-powered Web Research Agent built with **FastAPI**, **LangChain**, **OpenAI**, **ChromaDB**, **sentence-transformers**, and **Streamlit**.

This application can:

- Visit and scrape webpages
- Clean and process webpage content
- Generate embeddings
- Store knowledge in a vector database
- Answer questions using Retrieval-Augmented Generation (RAG)
- Maintain conversational memory
- Provide a modern chatbot interface

---

# Project Overview

The AI Web Research Agent is a production-ready NLP and AI engineering project designed to demonstrate modern Retrieval-Augmented Generation (RAG) architecture.

The application allows users to:

1. Input one or multiple URLs
2. Scrape webpage content
3. Clean HTML and extract readable text
4. Split content into semantic chunks
5. Generate embeddings using sentence-transformers
6. Store vectors in ChromaDB
7. Retrieve relevant context for questions
8. Answer with OpenAI models
9. Maintain conversational memory
10. Interact through a Streamlit chatbot UI

---

# Features

## Web Scraping

- Multi-URL support
- HTML cleaning
- Noise removal
- Retry mechanisms
- URL validation
- Error handling
- Basic caching

## RAG Pipeline

- Recursive text chunking
- Embedding generation
- Vector similarity search
- Context retrieval
- Grounded answer generation

## Conversational AI

- Chat history memory
- Context-aware conversations
- Multi-turn dialogue

## Frontend

- Streamlit chatbot interface
- Chat history
- Loading indicators
- Sidebar controls
- Session management

## Backend

- FastAPI REST API
- Modular architecture
- Scalable design
- Typed request/response models
- Logging system

---

# Architecture

```text
User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
Web Scraper → HTML Cleaner
  ↓
Chunking
  ↓
Embeddings
  ↓
ChromaDB Vector Store
  ↓
Retriever
  ↓
OpenAI LLM
  ↓
Answer Generation
```

---

# Tech Stack

## Backend

- Python 3.11+
- FastAPI

## AI / NLP

- LangChain
- OpenAI API

## Embeddings

- sentence-transformers

## Vector Database

- ChromaDB

## Scraping

- requests
- BeautifulSoup4

## Frontend

- Streamlit

## Environment

- python-dotenv

---

# Project Structure

```text
ai_web_agent/
│
├── app/
│   ├── main.py
│   ├── scraper.py
│   ├── rag_pipeline.py
│   ├── vector_store.py
│   ├── embeddings.py
│   ├── chatbot.py
│   ├── memory.py
│   ├── utils.py
│   └── config.py
│
├── frontend/
│   └── streamlit_app.py
│
├── data/
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── dockerfile
├── docker-compose.yml
└── setup.sh
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-web-research-agent.git
cd ai-web-research-agent
```

---

## 2. Create a Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Copy the example file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

# Running the Project

## Start the Backend

```bash
uvicorn app.main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

---

## Start the Frontend

In another terminal:

```bash
streamlit run frontend/streamlit_app.py

```

Frontend URL:

```text
http://localhost:8501
```

---

# API Endpoints

## Health Check

```http
GET /health
```

---

## Ingest URLs

```http
POST /ingest
```

Example:

```json
{
  "session_id": "demo-session",
  "urls": [
    "https://example.com",
    "https://example.org"
  ]
}
```

---

## Ask Questions

```http
POST /ask
```

Example:

```json
{
  "session_id": "demo-session",
  "question": "What is the webpage about?"
}
```

---

# How the RAG Pipeline Works

## Step 1 — Web Scraping

The scraper downloads webpage HTML using requests.

---

## Step 2 — HTML Cleaning

BeautifulSoup removes:

- scripts
- styles
- navigation
- footer
- forms
- noisy UI elements

---

## Step 3 — Text Chunking

The cleaned text is split into overlapping chunks.

This improves:

- retrieval quality
- semantic search
- answer grounding

---

## Step 4 — Embedding Generation

Each chunk is converted into a dense vector using sentence-transformers.

Semantic similarity becomes measurable in vector space.

---

## Step 5 — Vector Storage

Embeddings are stored inside ChromaDB.

Metadata stored includes:

- source URL
- page title
- chunk index

---

## Step 6 — Retrieval

When a user asks a question:

1. The question is embedded
2. ChromaDB searches similar vectors
3. Relevant chunks are returned

---

## Step 7 — Answer Generation

The retrieved context and chat memory are sent to the OpenAI model.

The LLM generates grounded answers.

---

# Conversational Memory

The project uses LangChain ConversationBufferMemory.

This allows:

- follow-up questions
- multi-turn conversations
- contextual dialogue

Each session maintains its own independent memory.

---

# Embeddings Explained

Embeddings transform text into numerical vectors.

Similar texts become close in vector space.

Example:

- “AI research”
- “machine learning studies”

These texts generate nearby vectors even if the wording differs.

This enables semantic search.

---

# Vector Databases Explained

Traditional databases search exact words.

Vector databases search semantic meaning.

ChromaDB stores embeddings and retrieves the closest vectors using similarity search.

This is what powers Retrieval-Augmented Generation.

---

# Docker Deployment

## Build and Run

```bash
docker compose up --build
```

---

## Backend

```text
http://localhost:8000
```

## Frontend

```text
http://localhost:8501
```

---

# Running Tests

```bash
pytest
```

---

# Screenshots

Add screenshots here for a professional GitHub portfolio.

```markdown
![Home](docs/screenshots/home.png)
![Chat](docs/screenshots/chat.png)
![Pipeline](docs/screenshots/pipeline.png)
```

---

# Future Improvements

Potential future upgrades:

- Autonomous agents
- Browser automation
- PDF support
- Multi-user authentication
- Streaming responses
- Citation highlighting
- Hybrid retrieval
- Knowledge graph integration
- Evaluation metrics dashboard
- Async scraping
- Multi-modal support
- Agent tool usage

---

# Troubleshooting

## OPENAI_API_KEY Missing

Ensure `.env` contains:

```env
OPENAI_API_KEY=your_key_here
```

---

## Backend Not Reachable

Verify FastAPI is running on port 8000.

---

## Module Import Errors

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

---

## Empty Scraping Results

Some websites block scraping or require JavaScript rendering.

---

# Why This Project Is Valuable

This project demonstrates:

- AI Engineering
- NLP Engineering
- RAG pipelines
- LLM integration
- Vector databases
- Backend API development
- Frontend engineering
- Docker deployment
- Production-ready Python architecture

Perfect for:

- AI internships
- NLP internships
- Data Science portfolios
- ML Engineering portfolios
- Software Engineering portfolios

---

# License

MIT License
