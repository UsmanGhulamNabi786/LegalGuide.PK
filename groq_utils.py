import os
from langchain_groq import ChatGroq






def initialize_groq():
    groq_api_key = os.getenv("GROQ_API_KEY")  # Fetch from environment variable
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing!")
    
    return ChatGroq(
        groq_api_key=groq_api_key,  # Pass the key to ChatGroq
    )
    
    # Initialize the Groq model
    return ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        groq_api_key=groq_api_key
    )
