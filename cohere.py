from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env.
COHERE_API_KEY=os.getenv("COHERE_API_KEY")
from cohere import Client
# from typing import ForwardRef
from pydantic import BaseModel
from retriever import retriever

from vectordb import process_pdf
vectorstore,store,child_splitter,parent_splitter,content=process_pdf("48lawsofpower.pdf")

retriever = retriever(vectorstore,store,child_splitter,parent_splitter,content)
def CohereRerank(retriever):
   co = Client(api_key = COHERE_API_KEY)

   class CustomCohereRerank(CohereRerank):
    class Config(BaseModel.Config):
      arbitrary_types_allowed = True

   CustomCohereRerank.update_forward_refs()

   compressor = CustomCohereRerank(client=co)

   compression_retriever = ContextualCompressionRetriever(
      base_compressor=compressor, base_retriever=retriever
)
   return compression_retriever
