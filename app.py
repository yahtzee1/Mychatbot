import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("google/gemma-1.1-2b-it")

def respond(
    message,
    history: list[tuple[str, str]],
    max_tokens
):
    messages = []

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=1024,
        stream=True
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.ChatInterface(respond, description="# Chat With AI faster than groq")

demo.launch()