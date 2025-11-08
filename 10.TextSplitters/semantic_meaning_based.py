############### Not Working ##################

from langchain_text_splitters  import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """
Cricket is the heartbeat of millions, uniting people with every cheer and chase.
The sea whispers freedom, its waves carrying endless dreams and depth.
Farmers sow hope beneath the sun, nurturing life with patience and faith.
Their sweat turns soil into gold, feeding the soul of the nation.

Terrorism is a shadow that darkens humanity’s progress and peace.
It spreads fear where dialogue and compassion should thrive.
Innocent lives become victims of hatred’s blind fury.
Only unity, justice, and understanding can defeat such darkness.
"""