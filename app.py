import gradio as gr
from generate import answer

def respond(message, history):
    result = answer(message)
    return result["answer"]

if __name__ == "__main__":
    gr.ChatInterface(respond).launch()