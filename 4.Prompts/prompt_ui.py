from re import template
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

template = load_prompt('4.Prompts\\template.json')

model=ChatOpenAI()
st.header('Reaserch Tool')

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

if st.button('Summarise'):
    chain = template | model
    result = chain.invoke({
        'paper_input' : paper_input,
        'style_input' : style_input, 
        'length_input' : length_input
    })

    print(chain)
    st.write(result.content)

