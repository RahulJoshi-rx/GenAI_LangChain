from langchain_community.document_loaders import  WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Wikipedia:Very_short_featured_articles")

docs = loader.load()
#print(len(docs))
#print(docs[0].page_content)
#print(docs[0].metadata)

model = ChatOpenAI()

prompt = PromptTemplate(
    template = 'Answer the following question - \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()


chain = prompt|model|parser
result = chain.invoke({'question':'What proposal has been made regarding the length of featured articles?', 'text':docs[0].page_content})
print(result)