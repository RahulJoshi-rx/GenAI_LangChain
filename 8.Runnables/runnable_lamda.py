from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

from langchain_core.runnables import RunnableLambda

##################################################################
def word_counter(textData):
    return len(textData.split())

runnable_word_counter = RunnableLambda(word_counter)

result = runnable_word_counter.invoke('Hello world')
#print(result)
##################################################################

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()
output_parser = StrOutputParser()

joke_gen_chain= RunnableSequence(prompt, model, output_parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count':RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result1 = final_chain.invoke({'topic':'AI'})
#print(result1)

final_result = """{} \nword count - {}""".format(result1['joke'], result1['word_count'])
print(final_result)