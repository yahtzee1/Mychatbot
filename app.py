import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the inference client with the specific model from Hugging Face.
client = InferenceClient("google/gemma-1.1-2b-it")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message="You are a friendly Chatbot.",
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    use_local_model=False,
):
   
    # Initialize history if it's None
    if history is None:
        history = []

    
    
    # API-based inference 
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""
    for message_chunk in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        if stop_inference:
            response = "Inference cancelled."
            yield history + [(message, response)]
            return
        if stop_inference:
            response = "Inference cancelled."
            break
        token = message_chunk.choices[0].delta.content
        response += token
        yield history + [(message, response)]  # Yield history + new response

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