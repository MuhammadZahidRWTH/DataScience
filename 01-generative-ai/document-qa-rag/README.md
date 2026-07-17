# 🧠 RAG Document Q&A With Groq and Llama 3

A Streamlit application that performs Retrieval-Augmented Generation (RAG) powered Q&A over PDF research papers using Groq's Llama 3 model and vector search with FAISS. Upload your research papers, embed them, and ask natural language questions to extract accurate answers based on your documents.

---

## 🚀 Features

- 🧾 Load multiple PDF research papers at once
- 🧠 Generate vector embeddings using OpenAI
- 🔎 Perform semantic search with FAISS
- 🤖 Use Groq's Llama 3 model for generating answers
- 🧷 RAG architecture: answers based only on your own documents
- 📄 Display source content from the reference documents

---

## 📦 Installation

Clone the repository and install dependencies:

git clone https://github.com/sarthakking5/RAG-Document-QA.git

cd your-repo

pip install -r requirements.txt


## ⚙️ Usage

1. Place your PDF files inside a folder named `research_paper/` in the project root.
2. Run the Streamlit app:
3. Click the **"Document Embedding"** button to build the vector database.
4. Enter a question in the input field to query your documents.
5. View the AI-generated answer and the supporting document excerpts.

## 🧪 Testing

Basic manual testing:

- Launch the app
- Load research papers into `research_paper/`
- Create vector embeddings using the button
- Ask questions based on the content

## 📁 Project Structure

```text
your-repo/
├── app.py              # Main Streamlit application
├── research_paper/     # Folder containing PDF files
├── requirements.txt    # Required Python packages
├── .env                # Environment variables (API keys)
└── README.md           # This file


```
---

## 🛠️ Tech Stack

- **Python 3**
- **Streamlit** - Web interface
- **LangChain** - Integration of LLMs and vector databases
- **FAISS** - Vector similarity search
- **OpenAI Embeddings** - Vector representation of document content
- **Groq (Llama 3)** - Large Language Model for Q&A
- **dotenv** - Environment variable loader

---

## 🤖 Model Info

- **Embedding Model**: `OpenAI Embeddings`
- **LLM**: `Llama3-8b-8192` (via Groq API)

