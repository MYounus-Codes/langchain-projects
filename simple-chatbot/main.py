from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key="your_google_api_key_here"
)

def chat_with_model(user_input):
    chat_history = []
    chat_history.append({"user": user_input})
    response = model.invoke(user_input)
    chat_history.append({"model": response.content})
    return response.content

if __name__ == "__main__":
    print("Welcome to the Simple Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = chat_with_model(user_input)
        print(f"Bot: {response}")
