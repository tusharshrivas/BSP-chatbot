Bhilai Steel Plant RAG Chatbot

An offline Retrieval-Augmented Generation (RAG) chatbot that answers queries using local PDF documents from the Bhilai Steel Plant.
It works fully offline once models are downloaded and included in the project zip.

✨ Features

📂 Process PDF reports (normal & scanned with OCR)

🔍 Semantic Search using ChromaDB & Sentence Transformers

💬 Chat Interfaces: CLI, Gradio Web UI, Streamlit Web UI

📖 Automatic Citations (filename + page number)

🖥️ Offline Mode – No internet needed after setup

⚡ Auto-detects new/modified/deleted PDFs and re-indexes

🏗️ Tech Stack

Python 3.9+

LangChain + ChromaDB (RAG)

Sentence Transformers – embeddings

Flan-T5 (small) – local LLM inference

PyPDF2 / pdfplumber / pytesseract + poppler – PDF parsing & OCR

Gradio / Streamlit – user interfaces

Watchdog – file monitoring

📂 Project Structure
bhilai-steel-chatbot/
│
├── documents/           # PDF knowledge base
├── vector_db/           # ChromaDB (auto-created after ingestion)
├── logs/                # Runtime logs
├── models/              # Local models (embedding + LLM) [already included in zip]
│
├── config.py            # Global configs (paths, models, settings)
├── utils.py             # Utility functions (PDF parsing, OCR, chunking)
├── ingest.py            # Document ingestion + embeddings
├── chatbot.py           # Chatbot (CLI, Gradio, Streamlit)
├── requirements.txt     # Python dependencies
└── README.md            # Documentation (this file)

⚙️ Installation (For Your Guide)
1️⃣ Extract the Project

Unzip the provided folder bhilai-steel-chatbot.zip to a location like:

C:\projects\bhilai-steel-chatbot

2️⃣ Install Python

Make sure Python 3.9+ is installed.
Check:

python --version

3️⃣ Create Virtual Environment

In PowerShell or CMD:

cd C:\projects\bhilai-steel-chatbot
python -m venv .venv
.venv\Scripts\activate

4️⃣ Install Dependencies
pip install -r requirements.txt


(All models are already included in the zip → no downloads required.)

🚀 Usage
1️⃣ Ingest PDFs

Put your documents in the documents/ folder. Then run:

python ingest.py


This will:

Extract text/tables (OCR if scanned PDFs)

Generate embeddings

Store in ChromaDB (inside vector_db/)

2️⃣ Run the Chatbot

▶ Command Line Interface (CLI):

python chatbot.py cli


▶ Gradio Web UI:

python chatbot.py gradio


Open: http://localhost:7860

▶ Streamlit Web UI:

streamlit run chatbot.py


Open: http://localhost:8501

🔄 Updating PDFs

Add/remove/modify files in documents/

Run python ingest.py again

Or leave ingest.py running → it will auto-detect changes

⚡ Offline Mode (No Telemetry)

This project is fully offline. To suppress telemetry:
In PyCharm or terminal, set environment variables:

Windows PowerShell:

set STREAMLIT_TELEMETRY=0
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1


(Already configured in project settings, so your guide usually won’t need to change anything.)

👨‍💻 Development Notes

Configs → config.py

Logs → logs/

Models → models/ (portable, already included)

Reset database → delete vector_db/

📌 Requirements

Python 3.9+

Tesseract OCR (optional, only for scanned PDFs)

Poppler (optional, for OCR-based table extraction)