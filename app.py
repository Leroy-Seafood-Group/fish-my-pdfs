#!/usr/bin/env python3

import gradio as gr
import os
from src.custom_chatbot import CustomChatbot
from src.custom_files import CustomFiles

template = """
You are a knowledge bot. Use the following pieces of context to answer the question at the end. 
Provide a detailed answer if possible.
Write at the end that the user is responsible for checking that the information provided is correct (in Norwegian).
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Helpful answer in Norwegian:
"""

# Get OpenAI API key, see the readme for more info
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')

# Create a ChatBot and CustomFiles instances
custom_files = CustomFiles()
chatbot_obj = CustomChatbot(template, custom_files)

# Define UI elements and interactions
with gr.Blocks() as blocks:
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"],
                value='gpt-3.5-turbo',
                interactive=True,
                label="Select model",
                info="Click on the 📁 or 🗄️ button if you want to change the model during the conversation."
            )
        with gr.Column(scale=0.05):
            k_search = gr.Number(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                precision=0,
                label='K search',
                interactive=True,
                info="Number of search results to retrieve from vector database."
            )

    with gr.Row():
        chat_ui = gr.Chatbot(elem_id="chatbot", height=300)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                show_label=False,
                lines=2,
                placeholder=("Please upload document(s) or use existing document store to use the chatbot"),
                interactive=False,
                container=False,
            )

    with gr.Row():
        with gr.Column(scale=0.1):
            db_button = gr.Button("🗄️ Use document store")
            db_button.click(
                chatbot_obj.get_vector_db,
                [model_dropdown, k_search],
                outputs=[input_text]
            )
        with gr.Column(scale=0.1):
            upload_button = gr.UploadButton(
                "📁 Upload PDF document(s)",
                file_types=[".pdf"],
                file_count="multiple"
            )
            upload_button.upload(
                chatbot_obj.upload_files,
                [upload_button, model_dropdown, k_search],
                outputs=[input_text]
            )

        with gr.Column(scale=0.1):
            clear_button = gr.Button("🗑️ Clear chat")
            clear_button.click(lambda: None, None, chat_ui, queue=False)

    input_text.submit(
        chatbot_obj.user,
        [input_text, chat_ui],
        [input_text, chat_ui]
    ).then(
        chatbot_obj.generate_response,
        inputs=chat_ui,
        outputs=chat_ui
    )

# Launch UI
blocks.queue()
blocks.launch()