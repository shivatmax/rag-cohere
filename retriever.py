
from langchain.retrievers import ParentDocumentRetriever

def retriever(vectorstore,store,child_splitter,parent_splitter,content):
   retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
   )

   retriever.add_documents(content,ids=None)
   return retriever

