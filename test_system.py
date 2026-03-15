"""
System testing script for Bhilai Steel Plant RAG Chatbot
Lightweight checks for imports and utilities.
"""
import sys
import tempfile
from pathlib import Path

def test_imports():
    print("Testing imports...")
    modules = ["chromadb", "sentence_transformers", "torch", "transformers", "PyPDF2", "pdfplumber", "PIL", "numpy", "pandas", "tqdm", "watchdog"]
    ok = True
    for m in modules:
        try:
            __import__(m)
            print(f"  ✅ {m}")
        except Exception as e:
            print(f"  ❌ {m}: {e}")
            ok = False
    return ok

def test_config_utils():
    try:
        import config
        from utils import clean_text, chunk_text
        print("  ✅ config and utils loaded")
        sample = "This   is   messy\ntext."
        print("  clean_text ->", clean_text(sample))
        chunks = chunk_text(" ".join(["word"]*200), chunk_size=50, overlap=10)
        print("  chunk_text ->", len(chunks), "chunks")
        return True
    except Exception as e:
        print("  ❌ config/utils error:", e)
        return False

def main():
    print("Running lightweight system tests...")
    ok1 = test_imports()
    ok2 = test_config_utils()
    if ok1 and ok2:
        print("All lightweight tests passed.")
        return 0
    print("Some tests failed. Fix issues above.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
