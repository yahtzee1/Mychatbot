import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the inference client with the specific model from Hugging Face.
client = InferenceClient("google/gemma-1.1-2b-it")

def answer_question(role, question):
    # Define default parameters for the language model.
    params = {
        "max_tokens": 150,
        "temperature": 0.7  # Default temperature
    }

    # Adjust parameters based on the role
    if role == 'Professor':
        params["temperature"] = 0.3  # More precise and deterministic
        params["max_tokens"] = 200  # Potentially more detailed responses
    elif role == 'Student':
        params["temperature"] = 0.7  # A balance between creativity and coherence
        params["max_tokens"] = 100  # Concise explanations
    elif role == "Don't Care":
        params["temperature"] = 1.0  # More creative and diverse responses

    # Modify the query based on the role and send it to the model.
    modified_query = f"{question}"
    response = client.chat_completion(
        messages=[{"role": "user", "content": modified_query}],
        **params
    )

    return response.choices[0].text

# Define the interface description and settings.
description = "# Interactive Chat with GEMMA-1.1-2B-IT\n### Enter your query below to receive a response from the model."

with gr.Blocks(css=".button {margin: 5px; width: 150px; height: 50px; font-size: 16px; border-radius: 5px;}") as demo:
    with gr.Row():
        # Set a default value for the role to ensure something is always selected.
        role = gr.Radio(choices=["Don't Care"], label="Select your role", type="index", value="Don't Care")
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