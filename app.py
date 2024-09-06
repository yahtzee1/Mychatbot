import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the inference client with the specific model from Hugging Face.
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def answer_question(question):
   

    # Prepare the messages for the model in the structured format
    messages = [{"role": "system", "content": "Query based on user role and input"}]
    messages.append({"role": "user", "content": question})

    response = ""
    # Stream the response from the model
    for message_chunk in client.chat_completion(
        messages,
        max_tokens=params[2048],
        stream=True
    ):
        # Here, you would handle stop conditions or interruptions (not implemented in this snippet)
        token = message_chunk.choices[0].delta.content  # Adjust to message_chunk.choices[0].text if delta.content is incorrect
        response += token
        yield response  # Yield response directly

# Define the interface description and settings.
description = "# Interactive Chat with GEMMA-1.1-2B-IT\n### Enter your query below to receive a response from the model."

with gr.Blocks(css=".button {margin: 5px; width: 150px; height: 50px; font-size: 16px; border-radius: 5px;}") as demo:
    with gr.Row():
        question = gr.Textbox(label="Enter your question")
    with gr.Row():
        button = gr.Button("Submit")
    output = gr.Textbox(label="Model Response")
    
    button.click(fn=answer_question, inputs=[question], outputs=output)

# Enable queuing to manage high demand if needed.
demo.queue(max_size=300000)

# Launch the Gradio web interface.
demo.launch()
