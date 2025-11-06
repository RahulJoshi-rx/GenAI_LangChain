from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.7)

prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

# Using LCEL (LangChain Expression Language) instead of deprecated LLMChain
chain = prompt | llm | parser

topic = input("Enter a topic : ")

output = chain.invoke({"topic": topic})

print("Generated blog title : ", output)