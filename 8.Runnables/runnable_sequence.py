from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()
output_parser = StrOutputParser()

chain1 = RunnableSequence(prompt1, model, output_parser)
result1 = chain1.invoke({'topic':'AI'})
#print(result1)

prompt2 = PromptTemplate(
    template='Explain the following joke {text}',
    input_variables=['text']
)

chain2 = RunnableSequence(prompt1, model, output_parser,prompt2, model, output_parser)
result2 = chain2.invoke({'topic':'AI'})
print(result2)