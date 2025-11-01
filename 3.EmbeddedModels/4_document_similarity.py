from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)
players_info = [
    "Virat Kohli is known for his aggressive batting style and unmatched consistency across formats.",
    "MS Dhoni, the former Indian captain, is celebrated for his calm leadership and finishing skills.",
    "Sachin Tendulkar, the 'Master Blaster', is regarded as one of the greatest batsmen in cricket history.",
    "Rohit Sharma is famous for his elegant stroke play and record-breaking double centuries in ODIs.",
    "Jasprit Bumrah is India's premier fast bowler, known for his deadly yorkers and control in death overs."
]

query = "tell me about bumrah"

doc_embeddings = embedding.embed_documents(players_info)
query_embedding = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
#print(similarity_scores)

index, score = sorted(list(enumerate(similarity_scores)),key=lambda x:x[1])[-1]
print(query)
print(players_info[index])
print("similarity score is : ", score)

top_k_indices = np.argsort(similarity_scores)[::-1][:3]

top_k_players = [players_info[i] for i in top_k_indices]

print(top_k_players)