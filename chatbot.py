# chatbot.py (replace the RAGChatbot class and main() with this block)
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import gradio as gr
import streamlit as st

from config import *
from utils import *


class RAGChatbot:
    def __init__(self):
        self.logger = setup_logging()
        self.logger.info("Initializing RAG Chatbot (offline mode)...")

        # embeddings (load from local path set in config.py)
        if not Path(EMBEDDING_MODEL).exists():
            raise FileNotFoundError(f"Embedding model folder not found at {EMBEDDING_MODEL}. Run download_models.py and copy models/ here.")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # chroma client
        try:
            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(CHROMA_DB_PATH))
            self.chroma_client = chromadb.Client(settings)
        except Exception:
            self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception:
            self.collection = self.chroma_client.create_collection(name=COLLECTION_NAME, metadata={"created": True})

        # tokenizer + model (load local files only)
        if not Path(LLM_MODEL).exists():
            raise FileNotFoundError(f"LLM model folder not found at {LLM_MODEL}. Run download_models.py and copy models/ here.")
        self.logger.info(f"Loading LLM model from local path: {LLM_MODEL}")
        # local_files_only True ensures HF won't try network if it is available/blocked
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL, local_files_only=True)

        # device placement
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")

    def embed_query(self, query: str):
        emb = self.embedding_model.encode([query], convert_to_numpy=True)
        return emb.tolist()

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, Dict]]:
        q_emb = self.embed_query(query)
        try:
            results = self.collection.query(
                query_embeddings=q_emb,
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            combined = list(zip(docs, metas))
            return combined
        except Exception as e:
            self.logger.error(f"ChromaDB query error: {e}")
            return []

    def build_prompt(self, user_query: str, contexts: List[Tuple[str, Dict]]) -> str:
        snippets = []
        for doc_text, meta in contexts:
            fn = meta.get("filename", "Unknown")
            pg = meta.get("page", "Unknown")
            snippet = doc_text[:2000]  # include more context per chunk
            snippets.append(f"[{fn} - Page {pg}]\n{snippet}")
        sources_block = "\n\n".join(snippets) if snippets else "No sources found."
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{sources_block}\n\nQuestion: {user_query}\n\nAnswer:"
        return prompt

    def generate_answer(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_RESPONSE_LENGTH,
                do_sample=False,
                temperature=float(TEMPERATURE)
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in text:
            return text.split("Answer:", 1)[-1].strip()
        return text.strip()

    def answer(self, user_query: str) -> Dict:
        contexts = self.retrieve(user_query, k=TOP_K_RETRIEVAL)
        prompt = self.build_prompt(user_query, contexts)
        response_text = self.generate_answer(prompt)

        # If the reply is too short, ask the model to expand (second pass)
        if len(response_text.split()) < 40:
            self.logger.info("Short answer detected — requesting expanded answer.")
            expand_prompt = prompt + "\n\nThe previous answer was short. Please provide a more detailed, step-by-step answer using only the context above. Include source citations as requested."
            response_text = self.generate_answer(expand_prompt)

        sources = [meta for _, meta in contexts]
        return {"text": response_text, "sources": sources}


# --- CLI / GUI runners ---
def run_cli(bot: RAGChatbot):
    print("Bhilai Steel Plant RAG Chatbot (type 'exit' to quit)")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        resp = bot.answer(q)
        print("\nAnswer:\n", resp["text"])
        print("\n" + format_citations(resp["sources"]))


def run_gradio(bot: RAGChatbot):
    def respond(query):
        out = bot.answer(query)
        return out["text"], format_citations(out["sources"])
    iface = gr.Interface(fn=respond, inputs=gr.Textbox(lines=3, placeholder="Ask a question..."), outputs=[gr.Textbox(), gr.Textbox()], title="Bhilai Steel Plant Chatbot")
    iface.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=False)


def run_streamlit(bot: RAGChatbot):
    st.set_page_config(page_title="Bhilai Steel Plant Chatbot")
    st.title("Bhilai Steel Plant — RAG Chatbot")
    query = st.text_input("Ask a question about internal documents:")
    if st.button("Ask") and query:
        with st.spinner("Retrieving and generating..."):
            out = bot.answer(query)
            st.subheader("Answer")
            st.write(out["text"])
            st.markdown("**Sources:**")
            st.write(format_citations(out["sources"]))


def main():
    parser = argparse.ArgumentParser()
    # change default to 'gradio' so running `python chatbot.py` opens GUI
    parser.add_argument("mode", nargs="?", default="gradio", choices=["cli", "gradio", "streamlit"], help="Mode to run the chatbot")
    args = parser.parse_args()
    bot = RAGChatbot()
    if args.mode == "cli":
        run_cli(bot)
    elif args.mode == "gradio":
        run_gradio(bot)
    elif args.mode == "streamlit":
        run_streamlit(bot)


if __name__ == "__main__":
    main()
