# Immigration Chatbot

A FastAPI-based immigration chatbot using OpenAI's API with streaming responses and secure endpoints.

## Features

- Real-time streaming responses
- Context-aware conversations
- Secure API endpoints with rate limiting
- PDF generation for immigration documents
- Efficient embedding caching
- Modern web interface

## Tech Stack

- FastAPI for the web server
- OpenAI API for chat completions and embeddings
- FAISS for vector similarity search
- ReportLab for PDF generation
- Slowapi for rate limiting
- JWT for API authentication

## Setup

1. Clone the repository:
```bash
git clone https://github.com/RegalApps/immigration-chatbot.git
cd immigration-chatbot
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and other settings
```

4. Run the server:
```bash
uvicorn immigration_chatbot:app --reload --port 8000
```

## API Documentation

- `/`: Web interface
- `/chat`: Chat endpoint (requires API key)
- `/download/{filename}`: Download generated PDFs (requires API key)

## Security

- API key authentication
- Rate limiting per endpoint
- CORS middleware
- Secure environment variable handling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
