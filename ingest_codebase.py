from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import os

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# file types that you want to ingest
extensions = ['.py', '.java', '.js', '.ts' , '.md']

docs = []

for dirpath, dirnames, filenames in os.walk(DATA_PATH):
    for file in filenames:
        try:
            _, file_extension = os.path.splitext(file)

            # skip file if it is not a file type that we want to ingest
            if file_extension not in extensions:
                continue

            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())

        except Exception as e:
            pass

text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)

# instead of saving it to a database, we can save it locally in a folder
db = FAISS.from_documents(texts, embeddings)
db.save_local(DB_FAISS_PATH)