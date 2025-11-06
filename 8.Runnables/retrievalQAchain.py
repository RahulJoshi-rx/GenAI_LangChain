from langchain_community.document_loaders import TextLoader  # For loading text files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting documents into chunks
from langchain_openai import OpenAIEmbeddings  # For creating embeddings using OpenAI
from langchain_community.vectorstores import FAISS  # For creating vector database using FAISS
from langchain_openai import OpenAI  # OpenAI LLM (not used in this code)
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# Load the text document from file
loader = TextLoader("8.Runnables/docs.txt")  # Create a loader for the text file
documents = loader.load()  # Load the document content into memory

# Split the document into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Create splitter with 500 char chunks, 50 char overlap
doc = text_splitter.split_documents(documents)  # Split the loaded documents into chunks

# Create a vector store from the document chunks
vectorstore = FAISS.from_documents(doc, OpenAIEmbeddings())  # Create FAISS vector store with OpenAI embeddings
retriever = vectorstore.as_retriever()  # Convert vector store to a retriever interface

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.7)  # Initialize the LLM model

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)  # Create a retrieval-based question-answering chain

query = "What are the key takeaways from the document?"  # Define the search query

answer = qa_chain.invoke({"query": query})  # Run the question-answering chain with the query and retrieve the answer
print(answer)