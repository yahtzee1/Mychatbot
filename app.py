import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("google/gemma-1.1-2b-it")

system_instructions = "[SYSTEM] Your task is to Answer the question. Keep conversation very short, clear and concise. The expectation is that you will avoid introductions and start answering the query directly, Only answer the question asked by user, Do not say unnecessary things.[QUESTION]"

def models(Query): 
    
    messages = []
    
    messages.append({"role": "user", "content": Query})

    Response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=2048,
        stream=True
    ):
        token = message.choices[0].delta.content

        Response += token
        yield Response

description="# Chat GO\n### Enter your query and Press enter and get response faster than groq"

demo = gr.Interface(description=description,fn=models, inputs=["text"], outputs="text")
demo.queue(max_size=300000)
demo.launch()
