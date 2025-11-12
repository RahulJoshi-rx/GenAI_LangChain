from importlib.resources import contents
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []
with open('4.Prompts\\chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())

print(chat_history)

prompt = chat_template.invoke({'query': 'What is the status of my order #12345?', 'chat_history': chat_history})
print(prompt)