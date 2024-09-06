import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the inference client with the specific model from Hugging Face.
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def models(query):
   

    # Prepare the messages for the model in the structured format
    messages = []
    messages.append({"role": "user", "content": f"[SYSTEM] You are ASSISTANT who answer question asked by user in short and concise manner. [USER] {query}"})

    response = ""
    
    for message_chunk in client.chat_completion(
        messages,
        max_tokens=params[2048],
        stream =True
    ):
        # Here, you would handle stop conditions or interruptions (not implemented in this snippet)
        token = message_chunk.choices[0].delta.content  # Adjust to message_chunk.choices[0].text if delta.content is incorrect
        response += token
        yield response  # Yield response directly

# Define the interface description and settings.
description = "# Interactive Chat with GEMMA-1.1-2B-IT\n### Enter your query below to receive a response from the model."

demo = gr.Interface(description=description,fn=models, inputs=["text"], outputs="text")

# Enable queuing to manage high demand if needed.
demo.queue(max_size=300000)

# Launch the Gradio web interface.
demo.launch()
