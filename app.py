import os
import gradio as gr
import requests

# Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-70b-8192"

class GroqClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def chat(self, messages: list):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        response = requests.post(self.endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}\n{response.text}"

# Initialize
groq_client = GroqClient(api_key=GROQ_API_KEY, model=MODEL)
chat_history = []

def chatbot(user_input):
    chat_history.append({"role": "user", "content": user_input})
    response = groq_client.chat(chat_history)
    chat_history.append({"role": "assistant", "content": response})
    return response

custom_css = """
body {
    background-color: #000;
    font-family: 'Segoe UI', sans-serif;
}
#chat-container {
    background: #111;
    color: #fff;
    border-radius: 18px;
    box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
}
#chat {
    background: transparent;
    padding: 24px 18px 12px 18px;
    height: 400px;
    overflow-y: auto;
    font-size: 1rem;
    color: #fff;
    border: none;
}
.user, .bot {
    margin-bottom: 14px;
    line-height: 1.5;
}
.user strong {
    color: #fff;
    font-weight: 600;
}
.bot strong {
    color: #bbb;
    font-weight: 600;
}
.user {
    text-align: right;
}
.bot {
    text-align: left;
}
#input-area {
    display: flex;
    border-top: 1px solid #222;
    background: #111;
    padding: 12px 18px;
}
#userInput {
    flex: 1;
    background: #222;
    border: none;
    border-radius: 8px;
    padding: 10px 14px;
    color: #fff;
    font-size: 1rem;
    outline: none;
    transition: background 0.2s;
}
#userInput:focus {
    background: #111;
}
#sendBtn {
    background: #fff;
    color: #000;
    border: none;
    border-radius: 8px;
    margin-left: 10px;
    padding: 10px 18px;
    font-size: 1rem;
    cursor: pointer;
    transition: background 0.2s;
}
#sendBtn:hover {
    background: #bbb;
    color: #000;
}
"""

# Gradio UI
with gr.Blocks(css=custom_css, title="ChatCore") as ui:
    gr.HTML("<h1 style='text-align: center;'>ChatCore</h1><p style='text-align: center;'>An AI chatbot powered by Groq and LLaMA 3, built with Gradio.</p>")
    with gr.Row():
        chatbot_input = gr.Textbox(label="Your Prompt", placeholder="Ask anything...", lines=2)
    output = gr.Textbox(label="Response", interactive=False)
    submit_btn = gr.Button("Send")

    def handle_input(text):
        return chatbot(text)

    submit_btn.click(fn=handle_input, inputs=[chatbot_input], outputs=[output])

ui.launch()
