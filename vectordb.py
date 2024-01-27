from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from LLM import embeddings

def process_pdf(file_path):
    data = UnstructuredPDFLoader(file_path)
    content = data.load()
    # print(content)
    # len(content[0].page_content)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    vectorstore = Chroma(embedding_function=embeddings)
    store = InMemoryStore()
    return vectorstore,store,child_splitter,parent_splitter,content



