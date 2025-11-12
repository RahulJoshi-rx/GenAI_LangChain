from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

#schema
class Review(BaseModel):
    
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str  = Field(description="A brief summary of review")
    sentiment: Literal["pos","neg"]= Field(description="Return sentiment of review wither positive, negative or neutral")
    pros: Optional[list[str]]= Field(default=None, description="Write down all the pros in a list")
    cons: Optional[list[str]]= Field(default=None, description="Write down all the cons in a list")
    name: Optional[str]= Field(default=None, description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Just got this phone last week and honestly it's pretty amazing! The screen is huge and super bright, even in sunlight. Camera quality is insane - took some pics at my friend's birthday party and they came out crystal clear even in low light.

The phone feels premium but it's kinda heavy ngl. Takes some getting used to if you're coming from a lighter phone. Battery easily lasts me the whole day with heavy use (lots of social media and gaming).

The S Pen is actually useful! I thought it would be a gimmick but I use it for quick notes and doodling. Gaming performance is smooth, no lag whatsoever even with demanding games.

Only complaints: it's expensive af and gets a bit warm when charging or gaming for long periods. Also it's pretty big so one-handed use is tough.

Overall: 9/10 would recommend if you have the budget. Best Android phone I've used so far!

Pros:

Amazing camera
Beautiful display
Great battery life
S Pen is actually useful
Cons:

Expensive
Heavy and bulky
Gets warm sometimes
Worth it if you want the best Android experience imo 👍

""")

print(type(result))
print(result)