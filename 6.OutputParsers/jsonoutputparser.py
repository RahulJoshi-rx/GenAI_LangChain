from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template|model|parser
result = chain.invoke({})
print(result)

template2 = PromptTemplate(
    template="Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain2 = template2|model|parser
result2 = chain2.invoke({'topic':"cricket"})
print(result2)


#prompt = template.invoke({})
#print(prompt)

#result = model.invoke(prompt)
#print(result)

#final_result = parser.parse(result.content)
#print(final_result)
#print(type(final_result))