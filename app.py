import gradio as gr
from huggingface_hub import InferenceClient


def client_fn(model):
    if "Nous" in model:
        return InferenceClient("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
    elif "Star" in model:
        return InferenceClient("HuggingFaceH4/starchat2-15b-v0.1")
    elif "Mistral" in model:
        return InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
    elif "Phi" in model:
        return InferenceClient("microsoft/Phi-3-mini-4k-instruct")
    else: 
        return InferenceClient("google/gemma-1.1-2b-it")

system_instructions1 = "[SYSTEM] Your task is to Answer the question. Keep conversation very short, clear and concise. The expectation is that you will avoid introductions and start answering the query directly, Only answer the question asked by user, Do not say unnecessary things.[QUESTION]"

def models(text, model="Mixtral 8x7B"): 
    
    client = client_fn(model)
    
    generate_kwargs = dict(
        max_new_tokens=100,
        do_sample=True,
    )
    
    formatted_prompt = system_instructions1 + text + "[ANSWER]"
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    for response in stream:        
        output+=response.token.text
        if output.endswith("<|assistant|>"):
            output = output[:-13]
        elif output.endswith("</s>"):
            output = output[:-4]
    return output

description="# Chat GO"

demo = gr.Interface(description=description,fn=models, inputs=["text", gr.Dropdown([ 'Mixtral 8x7B','Nous Hermes Mixtral 8x7B DPO','StarChat2 15b','Mistral 7B v0.3','Phi 3 mini', 'FASTEST' ], value="FASTEST", label="Select Model") ], outputs="text", live=True, batch=True, max_batch_size=10000)
demo.queue(max_size=300000)
demo.launch()
