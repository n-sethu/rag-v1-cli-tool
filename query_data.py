import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    print(f"Response: {response_text}\nSources: {sources}")
    return response_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    # Optional flag to populate DB before querying
    parser.add_argument("--populate", action="store_true", help="Populate DB before query.")
    args = parser.parse_args()

    if args.populate:
        import subprocess
        subprocess.run(["python3", "populate_database.py"])  # Modify path if needed

    query_rag(args.query_text)

if __name__ == "__main__":
    main()
