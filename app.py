import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = "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"
os.environ['HF_TOKEN'] = HF_TOKEN

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN
)
llm = ChatHuggingFace(llm=llm)

def getLLamaresponse(input_text, no_words, blog_style):
    if not input_text or not no_words:
        return "⚠️ Please enter a valid topic and word count."

    try:
        template = """
        Write a blog for {blog_style} job profile on the topic "{input_text}" 
        within {no_words} words.
        """

        prompt = ChatPromptTemplate.from_template(template)

        messages = [
            SystemMessage(content="You are a helpful blog writer."),
            HumanMessage(content=prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
        ]

        response = llm.invoke(messages)  
        return response.content if response else "⚠️ No response from the model."

    except Exception as e:
        return f"❌ Error: {str(e)}"


st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖")


input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', 
                              ('Researchers', 'Data Scientist', 'Common People'), 
                              index=0)

submit = st.button("Generate")
if submit:
    response = getLLamaresponse(input_text, no_words, blog_style)
    st.write(response)
