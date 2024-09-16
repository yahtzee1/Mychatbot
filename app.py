import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the inference client with the specific model from Hugging Face.
client = InferenceClient(model="google/gemma-1.1-7b-it")

def answer_question(role, question):
    # Define default parameters for the language model.
    params = {
        "max_tokens": 150,
        "temperature": 0.7  # Default temperature
    }

    # Adjust parameters based on the role
    if role == 'Professor':
        params["temperature"] = 0.2
        params["max_tokens"] = 250
    elif role == 'Student':
        params["temperature"] = 0.5
        params["max_tokens"] = 350
    elif role == "Don't Care":
        params["temperature"] = 1
        params["max_tokens"] = 600

    # Prepare the messages for the model without using 'system' role
    messages = [{"role": "user", "content": question}]
    
    try:
        # Call the model for chat completion
        response = client.chat_completion(
            messages=messages,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define the interface description and settings.
description = "# Interactive Chat with GEMMA-1.1-2B-IT\n### Enter your query below to receive a response from the model."

with gr.Blocks(css=".button {margin: 5px; width: 150px; height: 50px; font-size: 16px; border-radius: 5px;}") as demo:
    with gr.Row():
        role = gr.Radio(choices=["Professor", "Student", "Don't Care"], label="Select your role", value="Don't Care")
    with gr.Row():
        question = gr.Textbox(label="Enter your question")
    with gr.Row():
        button = gr.Button("Submit")
    output = gr.Textbox(label="Model Response")
    
    button.click(fn=answer_question, inputs=[role, question], outputs=output)

# Enable queuing to manage high demand if needed.
demo.queue(max_size=300000)

# Launch the Gradio web interface.
demo.launch()
