from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

#1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write detailed report on {topic}",
    input_variables=['topic']
)

#2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':"black hole"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

summaryResult = model.invoke(prompt2)

print(summaryResult.content)