# download_models.py
# Run once on a machine with internet. Saves models to ./models/
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("Downloading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb.save(str(models_dir / "all-MiniLM-L6-v2"))
print("Saved embedding model to: models/all-MiniLM-L6-v2")

print("Downloading LLM model (google/flan-t5-small)...")
tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tok.save_pretrained(str(models_dir / "flan-t5-small"))
model.save_pretrained(str(models_dir / "flan-t5-small"))
print("Saved LLM model to: models/flan-t5-small")

print("Done. Copy the ./models/ folder to your offline machine (project root).")
