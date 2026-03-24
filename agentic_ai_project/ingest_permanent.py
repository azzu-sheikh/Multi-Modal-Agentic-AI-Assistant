# Imports
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load env
load_dotenv()

# Main function
def ingest_data():
    # Load text
    loader = TextLoader("resume.txt")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings (models/ prefix required by Google API)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Build and save FAISS index
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_permanent")
    print("Ingestion complete. faiss_permanent saved.")

# Execute
if __name__ == "__main__":
    ingest_data()