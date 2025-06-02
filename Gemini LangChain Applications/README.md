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

🛠️ Setup Instructions

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⚙️ Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

🚀 Running the Applications

### Chatbot

```bash
streamlit run apps/chatbot_app.py
```

### Translator

```bash
streamlit run apps/translator_app.py
```

🏗️ Project Structure

```bash
gemini-langchain-applications/
├── .env                    # Environment variables
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── requirements.txt        # Dependency list
├── apps/                   # Application modules
│   ├── chatbot_app.py      # Chatbot implementation
│   └── translator_app.py   # Translation service
└── config.py               # Configuration loader
```

🛠 Troubleshooting

- **API Errors**: Ensure your Google API key is valid and quotas aren't exhausted.
- **Module Not Found**: Run `pip install -r requirements.txt`.
- **Streamlit Issues**: Try `streamlit cache clear`.


