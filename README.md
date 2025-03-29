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

## Branch Strategy

### Branch Types
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features or improvements
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent fixes for production

### Development Workflow
1. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   ```

3. Push your branch and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request on GitHub targeting the `develop` branch

### Branch Protection Rules
- All changes must be made through pull requests
- Pull requests require:
  - Passing CI checks
  - Code review approval
  - Up-to-date branch status
- Direct pushes to `main` and `develop` are prohibited

### Commit Message Format
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Adding or modifying tests
- `refactor:` Code changes that neither fix bugs nor add features
- `style:` Code style changes (formatting, etc.)
- `chore:` Maintenance tasks

Example: `feat: add clickable next prompts to chat interface`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
