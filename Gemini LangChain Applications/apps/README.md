# Gemini LangChain Applications


## Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Setup](#-setup-instructions)
- [Running Applications](#-running-the-applications)
- [Project Structure](#-project-structure)


## 🌟 Features
Two powerful applications built with Gemini AI:

1. **🤖 Chatbot Application**
   - Natural language question answering
   - Context-aware responses
   - Streamlit-based interactive UI

2. **🌍 Translator Application**
   - English to German translation
   - Real-time translation
   - Clean user interface

## 📋 Prerequisites
- Python 3.10+
- [Google AI API key](https://aistudio.google.com/apikey)
- Git (optional)

## 🛠️ Setup Instructions


1. Create Virtual Environment
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
4. Configure API Key
Create .env file:

env
GOOGLE_API_KEY=your_actual_api_key_here
🚀 Running the Applications
Chatbot
bash
streamlit run apps/chatbot_app.py
Translator
bash
streamlit run apps/translator_app.py
🏗️ Project Structure
Gemini LangChain Applications/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── apps/
│   ├── chatbot_app.py
│   └── translator_app.py
└── config.py

