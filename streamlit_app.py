import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from together import Together

load_dotenv()

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

st.title("Generate a sales pitch for any product🐦")

product = st.text_input("Product name")
target_audience = st.text_input("Target Audience")
key_features = st.text_input("Key Features")

persona = "You are a creative writer that writes sales pitches for a product."

instructions = """
            You need to generate a unique sales pitch for the provided product. 
            Ensure the pitch is engaging and tailored to the target audience, 
            highlighting the key features of the product. Make it persuasive and 
            relevant to the audience.
"""

prompt = """
            Create an engaging and persuasive sales pitch for the product based on 
            the provided product name, target audience, and key features. The pitch 
            should be tailored to the target audience and highlight the key features 
            of the product. Make it compelling and relevant.
    """

system_template = f"""
        {prompt}
        Respond in the persona of {persona}
        Respond in the instruction of {instructions}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define context messages based on user input
def create_context(product, target_audience, key_features):
    return [
        HumanMessage(content=f"Product: {product}"),
        HumanMessage(content=f"Target Audience: {target_audience}"),
        HumanMessage(content=f"Key Features: {key_features}"),
    ]

if st.button("Generate"):
    context = create_context(product, target_audience, key_features)
    
    # Create the prompt using the prompt template
    prompt_text = prompt_template.format(messages=context)
    
    # Send request to Together API
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt_text}]
    )
    
    # Initialize response content
    pitch_content = ""

    # Check if the response has 'choices' attribute
    if hasattr(response, 'choices'):
        for choice in response.choices:
            if hasattr(choice, 'message'):
                pitch_content += choice.message.content

    # Display the response content
    st.write(pitch_content)
