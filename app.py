import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("google/gemma-1.1-2b-it")

system_instructions = "[SYSTEM] Your task is to Answer the question. Keep conversation very short, clear and concise. The expectation is that you will avoid introductions and start answering the query directly, Only answer the question asked by user, Do not say unnecessary things.[QUESTION]"

def models(message): 
    
    messages = []
    
    messages.append({"role": "user", "content": f"[SYSTEM] You are ASSISTANT who answer question asked by user in short and concise manner. [USER] {message}"})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=200,
        stream=True
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

description="# Chat GO"

demo = gr.Interface(description=description,fn=models, inputs=["text"], outputs="text", batch=True, max_batch_size=10000)
demo.queue(max_size=300000)
demo.launch()
