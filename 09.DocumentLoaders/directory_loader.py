from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='9.DocumentLoaders\\PDFs',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

#docs=loader.load()
#print(len(docs))
#print(docs[2].page_content)
#print(docs[2].metadata)


#for document in docs:
#    print(document.metadata)

docs1=loader.lazy_load()

for document in docs1:
    print(document.metadata)