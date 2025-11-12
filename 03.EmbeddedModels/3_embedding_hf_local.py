from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
documents =[
    "Delhi is the capital of india",
    "Paris is the capital of france",
    "Kolkata is the capital of west bengal"
]

text = "Delhi is the capital of india"
result =embedding.embed_query(text)
print(str(result))