import argparse
import os
import shutil
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from PyPDF2 import PdfReader


CHROMA_PATH = "chroma"
DATA_PATH = "pdf_data"

MAX_CHUNKS_ALLOWED = 10000  # adjust as you see fit
def is_pdf_too_large(pdf_path, max_size_mb=10):
    size_bytes = os.path.getsize(pdf_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb > max_size_mb

def estimate_chunks(pdf_path, chunk_size=800):
    reader = PdfReader(pdf_path)
    total_text = ""
    for page in reader.pages:
        total_text += page.extract_text() or ""
    estimated_chunks = len(total_text) // chunk_size + 1
    return estimated_chunks

def can_load_pdf(pdf_path, db, max_chunks=MAX_CHUNKS_ALLOWED):
    # Estimate chunks by file size
    if is_pdf_too_large(pdf_path, max_size_mb=10):
        print(f"Skipping '{pdf_path}' because it exceeds file size limit.")
        return False
    
    estimated_chunks = estimate_chunks(pdf_path)
    existing_chunks = len(set(db.get(include=[])["ids"]))
    
    if existing_chunks + estimated_chunks > max_chunks:
        print(f"Skipping '{pdf_path}' because it would exceed max chunks limit ({max_chunks}).")
        return False
    return True


def select_pdfs():
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDFs found in folder '{DATA_PATH}'. Exiting.")
        exit(1)
    
    print("Available PDFs:")
    for i, pdf in enumerate(pdf_files):
        print(f"{i}: {pdf}")

    selected_indices = input("Enter comma-separated indices of PDFs to load (e.g. 0,2): ")
    indices = []
    for idx in selected_indices.split(","):
        idx = idx.strip()
        if idx.isdigit():
            idx_int = int(idx)
            if 0 <= idx_int < len(pdf_files):
                indices.append(idx_int)

    selected_files = [pdf_files[i] for i in indices]
    if not selected_files:
        print("No valid PDFs selected. Exiting.")
        exit(1)
    
    return [os.path.join(DATA_PATH, pdf) for pdf in selected_files]

def load_documents(selected_pdf_paths):
    """Load PDFs using multiple strategies with fallbacks.

    Tries, in order: UnstructuredPDFLoader, PDFium2Loader, PyPDFLoader (pypdf),
    and finally a minimal fallback using pdfminer.six's extract_text.
    Each loader is attempted only if its import succeeds. Any loader exceptions
    for a specific PDF are logged and the next loader is tried.
    """
    from typing import List
    docs: List = []

    for pdf_path in selected_pdf_paths:
        # Try UnstructuredPDFLoader (best quality for messy PDFs)
        try:
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(pdf_path, strategy="fast")
                loaded = loader.load()
                if loaded:
                    docs.extend(loaded)
                    continue
            except Exception:
                # import or runtime failure; fall through to next loader
                pass

            # Try PDFium2Loader (fast raster-based extractor)
            try:
                from langchain_community.document_loaders import PDFium2Loader
                loader = PDFium2Loader(pdf_path)
                loaded = loader.load()
                if loaded:
                    docs.extend(loaded)
                    continue
            except Exception:
                pass

            # Try PyPDFLoader (pypdf). This used to be the default but can be slow
            # on some PDFs; still useful as a middle-ground.
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                loaded = loader.load()
                if loaded:
                    docs.extend(loaded)
                    continue
            except Exception:
                pass

            # Final minimal fallback using pdfminer.six's extract_text
            try:
                from pdfminer.high_level import extract_text
                text = extract_text(pdf_path) or ""
                # lazy-create a minimal langchain Document
                try:
                    # prefer langchain.schema.Document if available
                    from langchain.schema import Document
                except Exception:
                    # older langchain_core installs expose a compatible Document
                    try:
                        from langchain_core.schema import Document
                    except Exception:
                        # last resort: create a simple dict-like object
                        Document = None

                if Document is not None:
                    docs.append(Document(page_content=text, metadata={"source": pdf_path}))
                else:
                    docs.append({"page_content": text, "metadata": {"source": pdf_path}})
                continue
            except Exception as e:
                print(f"pdfminer fallback failed for {pdf_path}: {e}")

            # If we reach here, all loaders failed for this pdf
            print(f"All PDF loaders failed for {pdf_path}; skipping.")

        except KeyboardInterrupt:
            # propagate ctrl-c so the script can be interrupted normally
            raise

    return docs
def split_documents(documents: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list):
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    
    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        
        chunk.metadata["id"] = chunk_id
    
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared database at '{CHROMA_PATH}'.")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    
    selected_pdf_paths = select_pdfs()
    documents = load_documents(selected_pdf_paths)
    chunks = split_documents(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Encountered error during db population:", e)
        print("Check installed PDF extractor packages and try again.")
