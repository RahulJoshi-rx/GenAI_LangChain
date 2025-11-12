# Import necessary libraries for document processing and retrieval
from langchain_community.document_loaders import TextLoader  # For loading text files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting documents into chunks
from langchain_openai import OpenAIEmbeddings  # For creating embeddings using OpenAI
from langchain_community.vectorstores import FAISS  # For creating vector database using FAISS
from langchain_openai import OpenAI  # OpenAI LLM (not used in this code)
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load the text document from file
loader = TextLoader("8.runnables\\docs.txt")  # Create a loader for the text file
documents = loader.load()  # Load the document content into memory

# Split the document into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Create splitter with 500 char chunks, 50 char overlap
doc = text_splitter.split_documents(documents)  # Split the loaded documents into chunks

# Create a vector store from the document chunks
vectorestore = FAISS.from_documents(doc, OpenAIEmbeddings())  # Create FAISS vector store with OpenAI embeddings
retriever = vectorestore.as_retriever()  # Convert vector store to a retriever interface

# Query the document and retrieve relevant chunks
query = "What are the key takeaways from the document?"  # Define the search query
retrieved_docs = retriever.invoke(query)  # Retrieve relevant document chunks based on the query

retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])  # Join the retrieved document chunks into a single string

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.7)  # Initialize the LLM model

prompt = PromptTemplate(
    template="Based on following text, answer the question : {query}\n\n{retrived_text}",
    input_variables=["query","retrived_text"]
)

answer = llm.invoke(prompt.invoke({'query': query, 'retrived_text': retrieved_text}))  # Invoke the LLM model with the prompt and retrieve the answer
print(answer)