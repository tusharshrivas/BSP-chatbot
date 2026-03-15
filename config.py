# config.py
from pathlib import Path
import os

# Project paths
PROJECT_ROOT = Path(__file__).parent.resolve()
PDF_FOLDER = PROJECT_ROOT / "documents"  # Folder containing PDF documents
CHROMA_DB_PATH = PROJECT_ROOT / "vector_db"
LOGS_PATH = PROJECT_ROOT / "logs"

# Ensure directories exist
PDF_FOLDER.mkdir(exist_ok=True)
CHROMA_DB_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)

# Force Hugging Face / Transformers offline mode (do this before any HF imports)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
# Optional caches inside project to avoid using user-level cache
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".cache" / "transformers"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

# Local models directory (where download_models.py saved them)
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDING_MODEL = str(LOCAL_MODELS_DIR / "all-MiniLM-L6-v2")   # local path
LLM_MODEL = str(LOCAL_MODELS_DIR / "flan-t5-small")           # local path

# ChromaDB configuration
COLLECTION_NAME = "bhilai_steel_docs"
CHUNK_SIZE = 800         # increase chunk size for more context (words)
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 8      # retrieve more docs for higher accuracy

# OCR configuration
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.07.0\Library\bin"

# File processing
SUPPORTED_EXTENSIONS = [".pdf"]
MAX_FILE_SIZE_MB = 200
BATCH_SIZE = 32

# Chatbot configuration
MAX_CONTEXT_LENGTH = 1024   # keep reasonable for small/medium LLMs
MAX_RESPONSE_LENGTH = 512   # increase response length
TEMPERATURE = 0.0
SYSTEM_PROMPT = """You are an AI assistant for Bhilai Steel Plant. Use ONLY the supplied context.
- If the context doesn't contain the answer, say: "I don't have sufficient information in the available documents to answer this question."
- Provide thorough, step-by-step answers. If multiple steps, number them.
- After each numbered paragraph include the source in parentheses, e.g. (filename.pdf — Page 4).
- Be factual and concise, but comprehensive."""
# UI configuration
STREAMLIT_PORT = 8501
GRADIO_PORT = 7860
# Monitoring
FILE_CHECK_INTERVAL = 30
# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
