
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from retriever import retriever
from prompt import template
from LLM import model

from vectordb import process_pdf
vectorstore,store,child_splitter,parent_splitter,content=process_pdf("48lawsofpower.pdf")

retriever = retriever(vectorstore,store,child_splitter,parent_splitter,content)

output_parser = StrOutputParser()
prompt = template()

chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)
print(chain.invoke("Can you tell me the story of Queen Elizabeth I from this 48 laws of power book?"))