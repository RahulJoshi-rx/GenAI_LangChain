from langchain_core.language_models import LLM
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, llms
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-genertion"
)

model=ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of india?")
print(result.content)
