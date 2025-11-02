from itertools import chain
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
#from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512
)

model1 = ChatOpenAI()
model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=['text']
)


prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['quiz', 'notes']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
    'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser,
    }
)

merge_chain = prompt3 |model1|parser

chain = parallel_chain | merge_chain

text = """Just got this phone last week and honestly it's pretty amazing! The screen is huge and super bright, even in sunlight. Camera quality is insane - took some pics at my friend's birthday party and they came out crystal clear even in low light.

The phone feels premium but it's kinda heavy ngl. Takes some getting used to if you're coming from a lighter phone. Battery easily lasts me the whole day with heavy use (lots of social media and gaming).

The S Pen is actually useful! I thought it would be a gimmick but I use it for quick notes and doodling. Gaming performance is smooth, no lag whatsoever even with demanding games.

Only complaints: it's expensive af and gets a bit warm when charging or gaming for long periods. Also it's pretty big so one-handed use is tough.

Overall: 9/10 would recommend if you have the budget. Best Android phone I've used so far!

Pros:

Amazing camera
Beautiful display
Great battery life
S Pen is actually useful
Cons:

Expensive
Heavy and bulky
Gets warm sometimes
Worth it if you want the best Android experience imo 👍

"""

result = chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii()
