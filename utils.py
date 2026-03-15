"""
Utility functions for the RAG chatbot
"""
import logging
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pdfplumber

from config import *

def setup_logging():
    log_file = LOGS_PATH / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger("bsp_rag")

logger = setup_logging()

def get_file_hash(file_path: Path) -> str:
    """Generate MD5 hash of file for change detection"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def clean_text(text: str) -> str:
    """Clean and preprocess extracted text"""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control chars except needed punctuation
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\/\%\$\@\#]', ' ', text)
    return ' '.join(text.split()).strip()

def extract_text_pypdf(pdf_path: Path) -> Dict[int, str]:
    """Extract text from PDF using PyPDF2 (PdfReader)"""
    text_by_page = {}
    try:
        with open(pdf_path, 'rb') as fh:
            reader = PdfReader(fh)
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    text_by_page[page_num] = clean_text(text)
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num} of {pdf_path}: {e}")
                    text_by_page[page_num] = ""
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
    return text_by_page

def extract_tables_pdfplumber(pdf_path: Path) -> Dict[int, List[str]]:
    """Extract tables from PDF using pdfplumber and return textual table representations per page"""
    tables_by_page = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = []
                try:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if table:
                                # Convert table to text representation
                                table_text = "\n".join([
                                    " | ".join([cell or "" for cell in row])
                                    for row in table if row
                                ])
                                page_tables.append(clean_text(table_text))
                except Exception as e:
                    logger.warning(f"Error extracting tables from page {page_num} of {pdf_path}: {e}")
                tables_by_page[page_num] = page_tables
    except Exception as e:
        logger.error(f"Error opening PDF for table extraction {pdf_path}: {e}")
    return tables_by_page

def extract_text_ocr(pdf_path: Path) -> Dict[int, str]:
    """Extract text from scanned PDF using OCR (pytesseract + pdf2image)"""
    text_by_page = {}
    try:
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        images = convert_from_path(str(pdf_path), dpi=300, poppler_path=POPPLER_PATH)
        for page_num, image in enumerate(images, start=1):
            try:
                text = pytesseract.image_to_string(image, lang='eng')
                text_by_page[page_num] = clean_text(text)
            except Exception as e:
                logger.warning(f"Error performing OCR on page {page_num} of {pdf_path}: {e}")
                text_by_page[page_num] = ""
    except Exception as e:
        logger.error(f"Error converting PDF to images for OCR {pdf_path}: {e}")
    return text_by_page

def is_scanned_pdf(pdf_path: Path, sample_pages: int = 3) -> bool:
    """Detect if PDF is scanned by checking text extraction success on first pages"""
    try:
        text_content = extract_text_pypdf(pdf_path)
        total_chars = 0
        pages_checked = 0
        for page_num, text in text_content.items():
            if pages_checked >= sample_pages:
                break
            total_chars += len(text.strip())
            pages_checked += 1
        avg_chars_per_page = total_chars / max(pages_checked, 1)
        return avg_chars_per_page < 50
    except Exception as e:
        logger.error(f"Error checking if {pdf_path} is scanned: {e}")
        return True  # conservative: treat as scanned

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks (by words)"""
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    step = chunk_size - overlap if chunk_size > overlap else chunk_size
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def load_file_metadata() -> Dict[str, Dict]:
    metadata_file = CHROMA_DB_PATH / "file_metadata.json"
    if metadata_file.exists():
        try:
            return json.load(open(metadata_file, "r", encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
    return {}

def save_file_metadata(metadata: Dict[str, Dict]):
    metadata_file = CHROMA_DB_PATH / "file_metadata.json"
    try:
        json.dump(metadata, open(metadata_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")

def format_citations(sources: List[Dict]) -> str:
    """Format document citations for display"""
    if not sources:
        return ""
    citations = []
    for source in sources:
        filename = source.get('filename', 'Unknown')
        page = source.get('page', 'Unknown')
        citations.append(f"{filename} (Page {page})")
    return "Sources: " + "; ".join(citations)

def validate_pdf_file(file_path: Path) -> bool:
    """Validate PDF (extension, size, readable)"""
    try:
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"File {file_path} exceeds size limit ({file_size_mb:.1f}MB)")
            return False
        # try reading basic structure
        with open(file_path, "rb") as fh:
            PdfReader(fh)
        return True
    except Exception as e:
        logger.error(f"Invalid PDF file {file_path}: {e}")
        return False
