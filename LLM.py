import os
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env.
HF_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key = HF_token,model_name = "thenlper/gte-large"
)
#intfloat/e5-mistral-7b-instruct

model = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",
                       model_kwargs={"temperature":0.1,"max_new_tokens":512,"max_length":64})





