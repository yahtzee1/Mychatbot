import gradio as gr
from huggingface_hub import InferenceClient


def client_fn(model):
    if "Mixtral" in model:
        return InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")
    elif "Llama" in model:
        return InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "Mistral" in model:
        return InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
    elif "Phi" in model:
        return InferenceClient("microsoft/Phi-3-mini-4k-instruct")
    else: 
        return InferenceClient("microsoft/Phi-3-mini-4k-instruct")

system_instructions1 = "[SYSTEM] Answer as Real Jarvis JARVIS, Made by 'Tony Stark', Keep conversation very short, clear, friendly and concise. The text provided is a request for a specific type of response from you, the virtual assistant. The request asks you to provide friendly responses as if You are the character Jarvis, made by 'Tony Stark.' The expectation is that I will avoid introductions and start answering the query directly, Only answer the question asked by user, Do not say unnecessary things.[USER]"

def models(text, model="Mixtral 8x7B"): 
    
    client = client_fn(model)
    
    generate_kwargs = dict(
        max_new_tokens=300,
    )
    
    formatted_prompt = system_instructions1 + text + "[JARVIS]"
    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=False, return_full_text=False)
    output = ""
    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text

    return output

demo = gr.Interface(fn=models, inputs=["text", gr.Dropdown([ 'Mixtral 8x7B','Llama 3 8B','Mistral 7B v0.3','Phi 3 mini', ], value="Mistral 7B v0.3", label="Select Model") ], outputs="text", live=True)
demo.launch()
