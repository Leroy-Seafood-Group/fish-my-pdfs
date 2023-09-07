# QA bot using local PDF files 
## Overview
This repository contains a Proof of Concept (PoC) for a QA chatbot capable of extracting and answering questions about information contained within PDF(s) using the GPT API. Frontend with Gradio. 

## Quick start 
<b>Clone the repository: </b>
'''bash
git clone https://github.com/Leroy-Seafood-Group/fish-my-pdfs.git
cd fish-my-pdfs
'''

<b>Install the required packages:</b> 
'''bash
pip install -r requirements.txt
'''

<b> ADD OpenAI API key </b>
see best practices for API key safety: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

<b> Run the Gradio app: </b>
'''bash
gradio app.py
'''

<b> Optional:</b>
Run the application in the notebook

Readings: https://scriv.ai/guides/retrieval-augmented-generation-overview/
