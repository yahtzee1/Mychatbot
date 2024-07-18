import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("google/gemma-1.1-2b-it")

def models(Query): 
    
    messages = []
    
    messages.append({"role": "user", "content": f"[SYSTEM] You are ASSISTANT who answer question asked by user in short and concise manner. [USER] {Query}"})

    Response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=2048,
        stream=True
    ):
        token = message.choices[0].delta.content

        Response += token
        yield Response

description="# Chat GO\n### Enter your query and Press enter and get lightning fast response"

demo = gr.Interface(description=description,fn=models, inputs=["text"], outputs="text")
demo.queue(max_size=300000)
demo.launch()