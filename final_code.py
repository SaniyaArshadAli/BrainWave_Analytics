import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Predefined responses
healthcare_responses = {
    "hello": "Hello! How can I assist you today?",
    "what is your name": "I am the Healthcare Chatbot. How can I help you?",
    "goodbye": "Goodbye! Take care.",
    # Add more predefined responses as needed
}

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    st.title("Healthcare Chatbot")

    st.write("Welcome to the Healthcare Chatbot! Type your message below:")

    while True:
        user_input = st.text_input("You:", "").lower()  # Convert user input to lowercase

        # Check if the user input has a predefined response
        if user_input in healthcare_responses:
            st.text(f"Chatbot: {healthcare_responses[user_input]}")
        elif user_input == "exit":
            break
        else:
            # If no predefined response is found, use the GPT-2 model
            prompt = f"User: {user_input}\nChatbot: "
            response = generate_response(prompt)
            st.text(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
