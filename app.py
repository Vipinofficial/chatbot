# app.py
import streamlit as st
from chatbot import Chatbot

# Initialize the chatbot
chatbot = Chatbot()

# Title of the app
st.title("Chatbot using DialoGPT")

# Placeholder for chat history
chat_history = st.container()

# User input
user_input = st.text_input("You:", "")

if user_input:
    # Generate response
    response = chatbot.generate_response(user_input)
    
    # Display conversation
    with chat_history:
        st.write(f"You: {user_input}")
        st.write(f"Bot: {response}")
