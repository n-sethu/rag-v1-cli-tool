# rag-v1-cli-tool

A Retrieval-Augmented Generation (RAG) CLI tool to build a Chroma vector database from selected PDFs and query it with an LLM (Ollama).  
This tool lets you **interactively select which PDFs to ingest** to avoid large database bloat and maintain control.

---

## Features

- Interactive PDF selection from `pdf_data/` folder — no bulk ingestion of all PDFs at once.  
- Splits PDFs into manageable chunks with overlap to improve context.  
- Stores chunk embeddings in Chroma vector database.  
- Query the vector DB with natural language questions using the Ollama LLM.  
- Simple CLI usage with reset option for the database.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/n-sethu/rag-v1-cli-tool.git
cd rag-v1-cli-tool
```

### 2. Create and activate Python env
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
Create a requirements.txt with:
```bash
langchain
langchain-community
PyPDF2
langchain-chroma
ollama
```

Install with 
```bash
pip install -r requirements.txt
```

Place any pdfs into a folder called /pdf_data 
#### make sure to place this folder in .gitignore so large files are not tracked
### Populate the vector database
```bash
python3 populate_database.py
```
- You will see a numbered list of PDFs inside pdf_data/.
- Enter comma-separated indices of PDFs you want to ingest (e.g., 0,2).
- The script will load, chunk, embed, and add only those PDFs to the Chroma vector database.
- To clear the database before populating, use the --reset flag:
```bash
python3 populate_database.py --reset
```
Query the vector database
```bash
python3 query_data.py "Your natural language question here"
```

# Customization
- Add custom PDF files and experiment with Ollama models. (You need ollama to run this, and I use Mistral (4.4 GB)
  
# License
This project is open source and available under the MIT License.

---

# Contact

Feel free to reach out or connect if you'd like to collaborate or discuss the work:

GitHub: @nikhil-sethu
LinkedIn: linkedin.com/in/nikhil-sethu/
