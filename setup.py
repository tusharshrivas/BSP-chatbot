"""
Setup script for Bhilai Steel Plant RAG Chatbot
"""
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing Python packages from requirements.txt ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_directories():
    print("Setting up directories...")
    for directory in ["documents", "vector_db", "logs"]:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_tesseract():
    print("Checking Tesseract OCR installation...")
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=True)
        print("✅ Tesseract OCR is available")
        print(result.stdout.splitlines()[0])
        return True
    except Exception:
        print("⚠️  Tesseract OCR not found. If you need OCR, install Tesseract.")
        return False

def create_sample_pdf():
    print("Creating sample PDF for testing...")
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        sample_pdf = Path("documents/sample_bhilai_document.pdf")
        c = canvas.Canvas(str(sample_pdf), pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, "BHILAI STEEL PLANT - INTERNAL DOCUMENT")
        c.drawString(100, 720, "Safety Guidelines and Procedures")
        c.drawString(100, 680, "1. All employees must wear safety helmets in production areas")
        c.showPage()
        c.drawString(100, 750, "BHILAI STEEL PLANT - QUALITY STANDARDS")
        c.save()
        print(f"✅ Sample PDF created: {sample_pdf}")
        return True
    except Exception:
        print("⚠️  reportlab not installed or failed; skip sample PDF creation")
        return False

def main():
    print("Bhilai Steel Plant RAG Chatbot - Setup")
    if not install_requirements():
        print("Setup failed during package installation.")
        sys.exit(1)
    setup_directories()
    check_tesseract()
    create_sample_pdf()
    print("\nSetup completed. Next steps:")
    print("1. Put PDF files into the 'documents' folder.")
    print("2. Run: python ingest.py")
    print("3. Run chatbot: python chatbot.py (CLI) OR python chatbot.py gradio OR python chatbot.py streamlit")

if __name__ == "__main__":
    main()
