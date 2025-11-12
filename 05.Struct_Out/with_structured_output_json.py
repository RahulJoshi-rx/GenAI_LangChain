from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()
model = ChatOpenAI()

#schema
json_schema = {
    "title":"Review",
    "type":"object",
    "properties":{
        "key_themes":{
            "type":"array",
            "items":{
                "type":"string"
            },
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary":{
            "type":"string",
            "description": "A brief summary of review"
        },
        "sentiment":{
            "type":"string",
            "enum" : ["pos", "neg"],
            "description": "Return sentiment of review wither positive, negative or neutral"
        },
        "pros":{
            "type":["array","null"],
            "items":{
                "type":"string"
            },
            "description": "Write down all the pros in a list"
        },
        "cons":{
            "type":["array","null"],
            "items":{
                "type":"string"
            },
            "description": "Write down all the cons in a list"
        },
        "name":{
            "type":["string","null"],
            "description": "Write the name of the reviewer"
        },
    },
    "required":["key_themes","summary","sentiment"]
}

structured_model = model.with_structured_output(json_schema)

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