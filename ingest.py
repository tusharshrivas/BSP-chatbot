"""
Document ingestion script for Bhilai Steel Plant RAG Chatbot
Processes PDFs, extracts text and tables, generates embeddings, and updates ChromaDB
"""
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tqdm import tqdm

from config import *
from utils import *


class DocumentProcessor:
    """Main document processor class"""

    def __init__(self):
        self.logger = setup_logging()
        self.logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        if not Path(EMBEDDING_MODEL).exists():
            raise FileNotFoundError(f"Embedding model folder not found at {EMBEDDING_MODEL}. Run download_models.py.")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Initialize ChromaDB client with persistent directory
        try:
            self.logger.info(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(CHROMA_DB_PATH))
            self.chroma_client = chromadb.Client(settings)
        except Exception:
            # fallback to simple PersistentClient if older chroma installed
            try:
                self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            except Exception as e:
                self.logger.error(f"Failed to initialize ChromaDB client: {e}")
                raise

        # Get or create collection
        try:
            try:
                self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
                self.logger.info(f"Loaded existing collection: {COLLECTION_NAME}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"created": datetime.now().isoformat()}
                )
                self.logger.info(f"Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            self.logger.error(f"Error obtaining collection: {e}")
            raise

        # File metadata for change detection
        self.file_metadata = load_file_metadata()

    def extract_document_content(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract all content from a PDF document"""
        self.logger.info(f"Processing document: {pdf_path}")
        content = {
            'filename': pdf_path.name,
            'filepath': str(pdf_path),
            'pages': {},
            'is_scanned': False
        }
        is_scanned = is_scanned_pdf(pdf_path)
        content['is_scanned'] = is_scanned

        if is_scanned:
            self.logger.info(f"Detected scanned PDF, using OCR: {pdf_path}")
            text_by_page = extract_text_ocr(pdf_path)
            tables_by_page = {}
        else:
            self.logger.info(f"Extracting text and tables: {pdf_path}")
            text_by_page = extract_text_pypdf(pdf_path)
            tables_by_page = extract_tables_pdfplumber(pdf_path)

        all_pages = set(text_by_page.keys()) | set(tables_by_page.keys())
        for page_num in all_pages:
            page_content = []
            if page_num in text_by_page and text_by_page[page_num]:
                page_content.append(text_by_page[page_num])
            if page_num in tables_by_page and tables_by_page[page_num]:
                for table in tables_by_page[page_num]:
                    page_content.append(f"TABLE:\n{table}")
            if page_content:
                content['pages'][page_num] = "\n\n".join(page_content)
        return content

    def create_document_chunks(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from document content with metadata"""
        chunks = []
        filename = content['filename']

        for page_num, page_content in content['pages'].items():
            if not page_content.strip():
                continue

            # FIX: avoid undefined variable errors by renaming to chunk_texts
            chunk_texts = chunk_text(page_content, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunk_texts:
                self.logger.warning(f"No chunks created for {filename} page {page_num}")
                continue

            for i, chunk_str in enumerate(chunk_texts):
                if not chunk_str.strip():
                    continue
                chunk_id = f"{filename}_page_{page_num}_chunk_{i}"
                chunk = {
                    'id': chunk_id,
                    'text': chunk_str,
                    'metadata': {
                        'filename': filename,
                        'filepath': content['filepath'],
                        'page': page_num,
                        'chunk_index': i,
                        'is_scanned': content['is_scanned'],
                        'timestamp': datetime.now().isoformat()
                    }
                }
                chunks.append(chunk)
        return chunks

    def generate_embeddings(self, texts: List[str]):
        """Generate embeddings for text chunks"""
        if not texts:
            return []
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def add_documents_to_db(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to ChromaDB"""
        if not chunks:
            return
        self.logger.info(f"Adding {len(chunks)} chunks to vector database")
        ids = [chunk['id'] for chunk in chunks]
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)

        # Add in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(chunks))
            batch_ids = ids[i:end]
            batch_texts = texts[i:end]
            batch_embeddings = embeddings[i:end]
            batch_metadatas = metadatas[i:end]
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                self.logger.error(f"Error adding batch to ChromaDB: {e}")

    def remove_document_from_db(self, filename: str):
        """Remove all chunks of a document from ChromaDB"""
        try:
            try:
                self.collection.delete(where={"filename": filename})
                self.logger.info(f"Deleted chunks for {filename} via metadata filter")
            except Exception:
                try:
                    results = self.collection.get(where={"filename": filename})
                    ids = results.get("ids", [])
                    if ids:
                        self.collection.delete(ids=ids)
                        self.logger.info(f"Removed {len(ids)} chunks for {filename}")
                except Exception as e:
                    self.logger.error(f"Fallback delete failed for {filename}: {e}")
        except Exception as e:
            self.logger.error(f"Error removing document {filename}: {e}")

    def process_single_document(self, pdf_path: Path) -> bool:
        """Process a single PDF document"""
        try:
            if not validate_pdf_file(pdf_path):
                self.logger.warning(f"Invalid PDF file: {pdf_path}")
                return False
            content = self.extract_document_content(pdf_path)
            if not content['pages']:
                self.logger.warning(f"No content extracted from: {pdf_path}")
                return False
            chunks = self.create_document_chunks(content)
            if not chunks:
                self.logger.warning(f"No chunks created from: {pdf_path}")
                return False
            self.remove_document_from_db(pdf_path.name)
            self.add_documents_to_db(chunks)
            file_hash = get_file_hash(pdf_path)
            self.file_metadata[str(pdf_path)] = {
                'hash': file_hash,
                'last_processed': datetime.now().isoformat(),
                'chunks_count': len(chunks)
            }
            save_file_metadata(self.file_metadata)
            self.logger.info(f"Successfully processed {pdf_path.name} ({len(chunks)} chunks)")
            return True
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return False

    def process_all_documents(self):
        """Process all PDF documents in the folder"""
        self.logger.info(f"Scanning for PDF files in: {PDF_FOLDER}")
        pdf_files = list(PDF_FOLDER.glob("**/*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        if not pdf_files:
            self.logger.warning("No PDF files found")
            return
        success_count = 0
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            if self.process_single_document(pdf_path):
                success_count += 1
        save_file_metadata(self.file_metadata)
        self.logger.info(f"Processing complete. {success_count}/{len(pdf_files)} files processed successfully")
        try:
            collection_count = self.collection.count()
            self.logger.info(f"Total chunks in database: {collection_count}")
        except Exception:
            pass

    def check_for_changes(self):
        """Check for new, modified, or deleted files"""
        current_files = set(PDF_FOLDER.glob("**/*.pdf"))
        processed_files = set(Path(p) for p in self.file_metadata.keys())
        changes_made = False
        for pdf_path in current_files:
            file_key = str(pdf_path)
            current_hash = get_file_hash(pdf_path)
            if (file_key not in self.file_metadata or self.file_metadata[file_key]['hash'] != current_hash):
                self.logger.info(f"Processing {'modified' if file_key in self.file_metadata else 'new'} file: {pdf_path.name}")
                if self.process_single_document(pdf_path):
                    changes_made = True
        deleted_files = processed_files - current_files
        for deleted_path in deleted_files:
            self.logger.info(f"Removing deleted file: {deleted_path.name}")
            self.remove_document_from_db(deleted_path.name)
            del self.file_metadata[str(deleted_path)]
            changes_made = True
        if changes_made:
            save_file_metadata(self.file_metadata)
            self.logger.info("Database updated with changes")
        return changes_made


class FileWatcher(FileSystemEventHandler):
    """File system event handler for monitoring PDF folder"""

    def __init__(self, processor: DocumentProcessor):
        self.processor = processor
        self.logger = processor.logger

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            time.sleep(1)
            pdf_path = Path(event.src_path)
            self.logger.info(f"New file detected: {pdf_path.name}")
            self.processor.process_single_document(pdf_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            time.sleep(1)
            pdf_path = Path(event.src_path)
            self.logger.info(f"Modified file detected: {pdf_path.name}")
            self.processor.process_single_document(pdf_path)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            pdf_path = Path(event.src_path)
            self.logger.info(f"Deleted file detected: {pdf_path.name}")
            self.processor.remove_document_from_db(pdf_path.name)
            file_key = str(pdf_path)
            if file_key in self.processor.file_metadata:
                del self.processor.file_metadata[file_key]
                save_file_metadata(self.processor.file_metadata)


def main():
    print("Bhilai Steel Plant - Document Ingestion System")
    print("=" * 50)
    processor = DocumentProcessor()
    print("\n1. Processing existing documents...")
    processor.process_all_documents()
    print(f"\n2. Starting file monitoring on: {PDF_FOLDER}")
    event_handler = FileWatcher(processor)
    observer = Observer()
    observer.schedule(event_handler, str(PDF_FOLDER), recursive=True)
    observer.start()
    try:
        print("\n3. Monitoring for changes... (Press Ctrl+C to stop)")
        while True:
            time.sleep(FILE_CHECK_INTERVAL)
            processor.check_for_changes()
    except KeyboardInterrupt:
        print("\n\nShutting down file monitoring...")
        observer.stop()
    observer.join()
    print("Document ingestion system stopped.")


if __name__ == "__main__":
    main()
