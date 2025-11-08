from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("9.DocumentLoaders\\cricket.txt", encoding='utf-8')
documents = loader.load()


#print(documents)
print(type(documents))
print(f"Loaded {len(documents)} document(s)")
print(type(documents[0]))

# Print document content
for i, doc in enumerate(documents):
    print(f"\nDocument {i+1}:")
#    print(f"Content: {doc.page_content[:200]}...")  # First 200 chars
    print(f"Metadata: {doc.metadata}")

model = ChatOpenAI()

prompt = PromptTemplate(
    template = 'Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()


chain = prompt|model|parser
result = chain.invoke({'poem':documents[0].page_content})
print(result)

