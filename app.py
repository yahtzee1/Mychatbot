import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the inference client with the specific model from Hugging Face.
client = InferenceClient("google/gemma-1.1-2b-it")

def models(query):
    # Prepare the messages in the structured format required by the model.
    messages = [{"role": "user", "content": f"{query}"}]
    
    # Try to get the response from the model and handle any errors that may occur.
    try:
        response = ""
        for message in client.chat_completion(
            messages,
            max_tokens=2048,
            stream=True
        ):
            # Append each token received from the streaming response to build the final response.
            token = message.choices[0].delta.content
            response += token
            yield response
    except Exception as e:
        # If there is an error during the inference, return an error message.
        yield f"An error occurred: {str(e)}"

# Define the interface description and settings.
description = "# Interactive Chat with GEMMA-1.1-2B-IT\n### Enter your query below to receive a response from the model."

# Setup the Gradio interface with appropriate configurations.
demo = gr.Interface(
    description=description,
    fn=models, 
    inputs=["text"], 
    outputs="text",
    examples=[["What is the weather like today?"]],
    allow_flagging="never",
    gr.Blocks(css=".button {margin: 5px; width: 150px; height: 50px; font-size: 16px; border-radius: 5px;}") as demo:
    with gr.Row():
        role = gr.Radio(choices=["Professor", "Student", "Don't Care"], label="Select your role", type="index")
    with gr.Row():
        question = gr.Textbox(label="Enter your question")
    with gr.Row():
        button = gr.Button("Submit")
    output = gr.Textbox(label="Model Response")
    
    button.click(fn=answer_question, inputs=[role, question], outputs=output)
)

# Enable queuing to manage high demand if needed.
demo.queue(max_size=300000)

# Launch the Gradio web interface.
demo.launch()
