from langchain_core import output_parsers
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()


prompt1 = PromptTemplate(
    template = 'write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Summarise the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()
output_parsers = StrOutputParser()


report_gen_chain = RunnableSequence(prompt1, model, output_parsers)
branch_chain = RunnableBranch(
    (lambda x:len(x.split())>200, RunnableSequence(prompt2, model, output_parsers)),
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_gen_chain, branch_chain)
result = final_chain.invoke({'topic':'Russia vs Ukraine'})
print(result)