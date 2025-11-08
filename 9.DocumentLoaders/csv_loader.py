from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("9.DocumentLoaders\\customers-100.csv")
documents = loader.load()

print(len(documents))
print(documents[0])