# Legal RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot for legal text analysis, specifically focused on US immigration law. The chatbot processes plain text input and leverages OpenAI's GPT models to provide immigration law insights and advice.

## Features

- Local text processing and chunking
- OpenAI embeddings for semantic search
- Context-aware responses with legal expertise
- Automated disclaimer generation for legal advice

## Setup

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install openai python-dotenv numpy pandas
```

3. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Add your immigration law text to `immigration.txt`

2. Run the chatbot:
```bash
python immigration_chatbot.py
```

The chatbot will:
- Process and chunk the input text
- Generate embeddings using OpenAI
- Provide relevant responses based on the legal context
- Include appropriate disclaimers with legal advice

## Note

This chatbot acts as a US Immigration lawyer with 10 years of experience in the tech industry. While it provides informed insights on immigration processes, visa requirements, and tech company immigration needs, all responses should be verified with a licensed legal professional.
