from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

passthrough = RunnablePassthrough()
#print(passthrough.invoke({'Name':'AI'}))


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

joke_gen_chain = RunnableSequence(prompt1, model, output_parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, model, output_parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result2 = final_chain.invoke({'topic':'cricket'})
print(result2)