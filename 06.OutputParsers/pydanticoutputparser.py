from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a data generator. You must respond ONLY with valid JSON. Do not include any explanations, code, markdown formatting, or additional text. Just return the raw JSON object."),
    ("user", """Generate the name, age and city of a fictional {place} person.

{format_instruction}

IMPORTANT: Your response must be ONLY the JSON object. Do not wrap it in markdown code blocks or add any other text.""")
])

# Use partial_variables to inject format instructions
template = template.partial(format_instruction=parser.get_format_instructions())

chain = template | model | parser
result = chain.invoke({'place': 'Srilankan'})
print(result)
