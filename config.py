from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# --- Sarvam API ---
SARVAM_OCR_MAX_PAGES_PER_JOB = 10  
SARVAM_SUMMARY_MODEL = "sarvam-105b"   
SARVAM_CHAT_MODEL = "sarvam-105b"    
SARVAM_API_TIMEOUT_SECONDS = 60
SARVAM_MAX_RETRIES = 3

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda"  

# --- Chunking ---
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50

# --- Retrieval ---
RETRIEVAL_TOP_K = 8
RERANK_TOP_N = 5 

# --- Vector store ---
CHROMA_COLLECTION_NAME = "govt_documents"

# --- Summarization accuracy check ---
SUMMARY_FACT_CHECK_ENABLED = True
