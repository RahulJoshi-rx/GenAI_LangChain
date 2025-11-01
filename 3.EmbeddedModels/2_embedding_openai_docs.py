from langchain_core import embeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
documents =[
    "Delhi is the capital of india",
    "Paris is the capital of france",
    "Kolkata is the capital of west bengal"
]

result =embedding.embed_documents(documents)
print(str(result))