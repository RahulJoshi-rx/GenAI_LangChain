from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("9.DocumentLoaders\\sample.pdf")
docs = loader.load()
#print(docs)
print(len(docs))
print(docs[2].page_content)
print(docs[2].metadata)
