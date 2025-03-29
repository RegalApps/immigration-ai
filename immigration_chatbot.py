# -*- coding: utf-8 -*-

import os
import sys
from typing import List, Dict, Optional, Any, Union, Tuple
import json
import time
import signal
import socket
import asyncio
import random
import numpy as np
from datetime import datetime, timedelta
import hashlib
import re

# OpenAI imports
from openai import OpenAI

# FastAPI imports
from fastapi import FastAPI, Request, Security, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
MAX_CACHE_SIZE = 1000  # Maximum number of items in each cache
CACHE_TTL = 3600  # Default cache TTL in seconds
RESPONSE_CACHE_TTL = 300  # Response cache TTL in seconds
CHUNK_SIZE = 1500  # Default chunk size for text processing

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize conversation history storage
conversation_histories = {}

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up API key authentication
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key."""
    if api_key == "test_key_123":  # This should match the key in chat.html
        return api_key
    raise HTTPException(
        status_code=403,
        detail="Could not validate API key"
    )

import logging
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Global server config
config = {
    'server': None  # Will store the Uvicorn server instance
}

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    print("\nShutting down server...")
    if config['server']:
        config['server'].should_exit = True
    sys.exit(0)

# Available functions for the chatbot
AVAILABLE_FUNCTIONS = {
    "create_timeline": {
        "name": "create_timeline",
        "description": "Create a timeline for immigration process",
        "parameters": {
            "type": "object",
            "properties": {
                "visa_type": {"type": "string"},
                "start_date": {"type": "string"},
                "key_events": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            },
            "required": ["visa_type", "start_date"]
        }
    },
    "save_case_details": {
        "name": "save_case_details",
        "description": "Save or update case information",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "details": {"type": "object"},
                "documents": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["case_id"]
        }
    },
    "generate_checklist": {
        "name": "generate_checklist",
        "description": "Create a document checklist",
        "parameters": {
            "type": "object",
            "properties": {
                "visa_type": {"type": "string"},
                "applicant_type": {"type": "string"}
            },
            "required": ["visa_type"]
        }
    },
    "fetch_uscis_form": {
        "name": "fetch_uscis_form",
        "description": "Get information about USCIS forms",
        "parameters": {
            "type": "object",
            "properties": {
                "form_number": {"type": "string"}
            },
            "required": ["form_number"]
        }
    },
    "calculate_dates": {
        "name": "calculate_dates",
        "description": "Calculate important immigration dates",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "visa_type": {"type": "string"}
            },
            "required": ["start_date", "visa_type"]
        }
    },
    "generate_pdf_summary": {
        "name": "generate_pdf_summary",
        "description": "Generate a PDF summary of immigration advice",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "advice_content": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "recommendations": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "timeline": {
                            "type": "array",
                            "items": {"type": "object"}
                        },
                        "required_documents": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["summary"]
                }
            },
            "required": ["case_id", "advice_content"]
        }
    }
}

async def handle_function_call(function_name: str, parameters: dict) -> dict:
    """Handle various function calls from Natalie"""
    if function_name == "create_timeline":
        # Create and return a timeline based on visa type
        events = []
        for event in parameters["key_events"]:
            events.append({
                "date": event["date"],
                "description": event["description"],
                "action_required": event.get("action_required", False)
            })
        return {"timeline": events}
        
    elif function_name == "save_case_details":
        case_id = parameters["case_id"]
        if case_id not in active_cases:
            active_cases[case_id] = ImmigrationCase(case_id)
        active_cases[case_id].update_details(parameters["details"])
        return {"status": "saved", "case_id": case_id}
        
    elif function_name == "generate_checklist":
        # Generate document checklist based on visa type
        visa_type = parameters["visa_type"]
        applicant_type = parameters["applicant_type"]
        # This would be expanded with actual visa requirements
        checklist = {
            "required_documents": [
                "Valid Passport",
                "Completed Form I-129",
                "Educational Credentials",
                "Employment Letter"
            ],
            "optional_documents": [
                "Previous Visa Documents",
                "Tax Returns"
            ]
        }
        return checklist
        
    elif function_name == "fetch_uscis_form":
        # Fetch USCIS form information
        form_info = {
            "I-129": {
                "title": "Petition for Nonimmigrant Worker",
                "url": "https://www.uscis.gov/i-129",
                "filing_fee": "$460"
            }
        }
        return form_info.get(parameters["form_number"], {})
        
    elif function_name == "calculate_dates":
        # Calculate important dates
        current_date = datetime.now()
        visa_expiry = datetime.strptime(parameters["start_date"], "%Y-%m-%d")
        
        dates = {
            "days_until_expiry": (visa_expiry - current_date).days,
            "recommended_renewal_date": (visa_expiry - timedelta(days=180)).strftime("%Y-%m-%d"),
            "latest_renewal_date": (visa_expiry - timedelta(days=45)).strftime("%Y-%m-%d")
        }
        return dates
        
    elif function_name == "generate_pdf_summary":
        # Create directory for PDFs if it doesn't exist
        os.makedirs("generated_pdfs", exist_ok=True)
        
        filename = f"generated_pdfs/immigration_advice_{parameters['case_id']}_{int(time.time())}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#2E5894')
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8
        )
        
        # Build document content
        content = []
        
        # Add header
        content.append(Paragraph("Immigration Advice Summary", title_style))
        content.append(Paragraph(f"Case ID: {parameters['case_id']}", heading_style))
        content.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", body_style))
        content.append(Spacer(1, 20))
        
        # Add main content sections
        if 'summary' in parameters["advice_content"]:
            content.append(Paragraph("Summary", heading_style))
            content.append(Paragraph(parameters["advice_content"]['summary'], body_style))
            content.append(Spacer(1, 15))
        
        if 'recommendations' in parameters["advice_content"]:
            content.append(Paragraph("Recommendations", heading_style))
            for idx, rec in enumerate(parameters["advice_content"]['recommendations'], 1):
                content.append(Paragraph(f"{idx}. {rec}", body_style))
            content.append(Spacer(1, 15))
        
        if 'timeline' in parameters["advice_content"]:
            content.append(Paragraph("Timeline", heading_style))
            timeline_data = [["Date", "Event", "Action Required"]]
            for event in parameters["advice_content"]['timeline']:
                timeline_data.append([
                    event['date'],
                    event['description'],
                    "Yes" if event.get('action_required') else "No"
                ])
            
            table = Table(timeline_data, colWidths=[1.5*inch, 4*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5894')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            content.append(table)
            content.append(Spacer(1, 15))
        
        if 'required_documents' in parameters["advice_content"]:
            content.append(Paragraph("Required Documents", heading_style))
            for doc in parameters["advice_content"]['required_documents']:
                content.append(Paragraph(f"• {doc}", body_style))
            content.append(Spacer(1, 15))
        
        # Add footer
        content.append(Spacer(1, 20))
        content.append(Paragraph(
            "This document was generated by Natalie, your immigration assistant. "
            "Please consult with a licensed immigration attorney for legal advice.",
            ParagraphStyle('Footer', parent=styles['Italic'], textColor=colors.gray)
        ))
        
        # Build PDF
        doc.build(content)
        return {"filename": filename}
        
    return {"error": "Function not found"}

async def format_citation(source_type: str, reference: dict) -> str:
    """Format a citation with proper legal reference."""
    source = IMMIGRATION_SOURCES.get(source_type)
    if not source:
        return ""
    
    citation = source["citation_format"].format(**reference)
    return f"{citation} ({source['url']})"

async def generate_advice_pdf(case_id: str, advice_content: dict) -> str:
    """Generate a professional PDF summary of immigration advice with citations."""
    filename = f"generated_pdfs/immigration_advice_{case_id}_{int(time.time())}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#2E5894')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8
    )
    
    citation_style = ParagraphStyle(
        'Citation',
        parent=styles['Italic'],
        fontSize=9,
        textColor=colors.gray,
        leftIndent=20
    )
    
    # Build document content
    content = []
    
    # Add header
    content.append(Paragraph("Immigration Advice Summary", title_style))
    content.append(Paragraph(f"Case ID: {case_id}", heading_style))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", body_style))
    content.append(Spacer(1, 20))
    
    # Add main content sections with citations
    if 'summary' in advice_content:
        content.append(Paragraph("Summary", heading_style))
        content.append(Paragraph(advice_content['summary'], body_style))
        if 'citations' in advice_content:
            for citation in advice_content['citations']:
                formatted_citation = await format_citation(
                    citation['source'],
                    citation['reference']
                )
                content.append(Paragraph(f"Source: {formatted_citation}", citation_style))
        content.append(Spacer(1, 15))
    
    if 'recommendations' in advice_content:
        content.append(Paragraph("Recommendations", heading_style))
        for idx, rec in enumerate(advice_content['recommendations'], 1):
            content.append(Paragraph(f"{idx}. {rec['text']}", body_style))
            if 'citation' in rec:
                formatted_citation = await format_citation(
                    rec['citation']['source'],
                    rec['citation']['reference']
                )
                content.append(Paragraph(f"Source: {formatted_citation}", citation_style))
        content.append(Spacer(1, 15))
    
    if 'timeline' in advice_content:
        content.append(Paragraph("Timeline", heading_style))
        timeline_data = [["Date", "Event", "Action Required", "Authority"]]
        for event in advice_content['timeline']:
            citation = ""
            if 'citation' in event:
                citation = await format_citation(
                    event['citation']['source'],
                    event['citation']['reference']
                )
            timeline_data.append([
                event['date'],
                event['description'],
                "Yes" if event.get('action_required') else "No",
                citation
            ])
        
        table = Table(timeline_data, colWidths=[1*inch, 3*inch, 1*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5894')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        content.append(table)
        content.append(Spacer(1, 15))
    
    if 'required_documents' in advice_content:
        content.append(Paragraph("Required Documents", heading_style))
        for doc in advice_content['required_documents']:
            content.append(Paragraph(f"• {doc['text']}", body_style))
            if 'citation' in doc:
                formatted_citation = await format_citation(
                    doc['citation']['source'],
                    doc['citation']['reference']
                )
                content.append(Paragraph(f"Required under: {formatted_citation}", citation_style))
        content.append(Spacer(1, 15))
    
    # Add Sources section
    content.append(Paragraph("Sources and References", heading_style))
    content.append(Paragraph(
        "This advice is based on the following official sources:",
        body_style
    ))
    for source in IMMIGRATION_SOURCES.values():
        content.append(Paragraph(
            f"• {source['name']}: {source['url']}",
            body_style
        ))
    content.append(Spacer(1, 15))
    
    # Add footer with disclaimer
    content.append(Spacer(1, 20))
    content.append(Paragraph(
        "This document was generated by Natalie, your immigration assistant. "
        "While every effort has been made to ensure accuracy and cite official sources, "
        "please consult with a licensed immigration attorney for legal advice. "
        "Immigration laws and policies may change - verify current requirements at USCIS.gov.",
        ParagraphStyle('Footer', parent=styles['Italic'], textColor=colors.gray)
    ))
    
    # Build PDF
    doc.build(content)
    return filename

def stable_hash(text: str) -> str:
    """Create a stable hash for a string."""
    return hashlib.sha256(text.encode()).hexdigest()

async def get_embedding(text: str, model="text-embedding-ada-002"):
    """Get embedding for a text using OpenAI's API."""
    text = text.replace("\n", " ")
    response = await client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

async def get_relevant_context(query: str, chunks: List[str] = None) -> str:
    """Get relevant context for a query using semantic search."""
    try:
        # Get chunks if not provided
        if chunks is None:
            with open("immigration.txt", "r", encoding="utf-8") as f:
                text = f.read()
            chunks = await get_cached_chunks(text)
        
        # Get query embedding
        query_embedding = await get_embedding(query)
        
        # Get chunk embeddings
        chunk_embeddings = await get_cached_embeddings(chunks)
        
        # Calculate similarities
        similarities = [
            cosine_similarity(query_embedding, chunk_embedding)
            for chunk_embedding in chunk_embeddings
        ]
        
        # Get top chunks
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
        relevant_chunks = [chunks[i] for i in top_indices]
        
        return "\n".join(relevant_chunks)
    except Exception as e:
        print(f"Error getting context: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks using simple sentence boundaries."""
    # First split by double newlines to preserve paragraph structure
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        # Split paragraph into sentences (simple approach)
        sentences = [s.strip() + '.' for s in para.replace('\n', ' ').split('.') if s.strip()]
        
        for sentence in sentences:
            if current_size + len(sentence) > chunk_size and current_chunk:
                # Join current chunk and add to chunks
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += len(sentence)
    
    # Add remaining chunk if any
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def get_cached_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Get chunks with caching."""
    cache_key = stable_hash(f"{text}_{chunk_size}")
    chunks = chunk_cache.get(cache_key)
    if chunks is None:
        chunks = chunk_text(text, chunk_size)
        chunk_cache.set(cache_key, chunks)
    return chunks

async def get_cached_embeddings(chunks: List[str]) -> List[List[float]]:
    """Get embeddings with caching and simple batching."""
    embeddings = []
    to_process = []
    indices = []

    # Check cache first
    for i, chunk in enumerate(chunks):
        chunk_key = stable_hash(chunk)
        cached = embedding_cache.get(chunk_key)
        if cached:
            embeddings.append(cached)
        else:
            to_process.append(chunk)
            indices.append(i)
    
    # Process uncached chunks in small batches
    if to_process:
        batch_size = 5  # Small batch size for better streaming
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i + batch_size]
            response = await client.embeddings.create(
                model="text-embedding-ada-002",  # Use consistent model
                input=batch
            )
            
            # Cache and insert embeddings
            batch_embeddings = [e.embedding for e in response.data]
            for j, emb in zip(range(i, min(i + batch_size, len(to_process))), batch_embeddings):
                chunk_key = stable_hash(to_process[j])
                embedding_cache.set(chunk_key, emb)
                embeddings.insert(indices[j], emb)
    
    return embeddings

async def get_chat_response(message: str, conversation_id: str) -> Tuple[str, List[str]]:
    """Get a response from the chatbot."""
    try:
        # Get relevant context
        context = await get_relevant_context(message)

        # Get or initialize conversation history
        if conversation_id not in conversation_histories:
            conversation_histories[conversation_id] = [
                {"role": "system", "content": """You are Natalie Chen, a warm and approachable immigration lawyer with 10 years of experience, particularly in tech industry immigration. You have a JD from Stanford Law and previously worked at a top Silicon Valley law firm before starting your own practice.

Your communication style:
- Friendly and empathetic, but always professional
- Use clear, everyday language instead of complex legal jargon
- Break down complex concepts into digestible pieces
- Acknowledge concerns and uncertainties
- Offer practical next steps and guidance
- Use occasional friendly phrases like "I understand this can feel overwhelming" or "Let me help you navigate this"
- Share relevant anecdotes from your experience when appropriate (without naming names)
- Always maintain accuracy and compliance with immigration law

Remember to:
1. Start responses with a brief acknowledgment of the question
2. Provide accurate information based on the context
3. Break down complex topics into clear steps
4. End with an encouraging note and invitation for follow-up questions
5. If something is unclear, ask for clarification rather than making assumptions
6. Always provide 3 relevant follow-up questions or prompts at the end of your response"""}
            ]

        # Add user message to history
        conversation_histories[conversation_id].append(
            {"role": "user", "content": f"Based on this context:\n\n{context}\n\nQuestion: {message}\n\nAfter your response, suggest 3 relevant follow-up questions or prompts that would help guide the conversation forward."}
        )

        # Get completion from OpenAI
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=conversation_histories[conversation_id],
            temperature=0.7,
            max_tokens=1000
        )

        # Extract response and next prompts
        full_response = completion.choices[0].message.content
        
        # Split the response into main content and next prompts
        parts = full_response.split("\n\n")
        main_response = parts[0]
        next_prompts = []
        
        # Look for numbered questions/prompts in the remaining parts
        for part in parts[1:]:
            if re.search(r'^\d+[\)\.] ', part):
                prompts = re.findall(r'^\d+[\)\.] (.+)$', part, re.MULTILINE)
                next_prompts.extend(prompts)
                if len(next_prompts) >= 3:
                    next_prompts = next_prompts[:3]
                    break

        # If no prompts found in the structured format, generate them
        if len(next_prompts) < 3:
            follow_up_completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are helping to generate 3 relevant follow-up questions based on the previous response. Make them specific and actionable."},
                    {"role": "user", "content": f"Based on this response:\n\n{main_response}\n\nGenerate 3 relevant follow-up questions that would help guide the conversation forward."}
                ],
                temperature=0.7,
                max_tokens=200
            )
            next_prompts = re.findall(r'^\d+[\)\.] (.+)$', follow_up_completion.choices[0].message.content, re.MULTILINE)[:3]

        # Add assistant's response to history
        conversation_histories[conversation_id].append(
            {"role": "assistant", "content": main_response}
        )

        # Trim history if it gets too long (keep last N messages)
        if len(conversation_histories[conversation_id]) > 10:
            conversation_histories[conversation_id] = (
                [conversation_histories[conversation_id][0]] +  # Keep system message
                conversation_histories[conversation_id][-9:]     # Keep last 9 messages
            )

        return main_response, next_prompts

    except Exception as e:
        return f"I apologize, but I'm having some technical difficulties at the moment. Could you please repeat your question? Error details: {str(e)}", []

async def stream_response(response_text: str, next_prompts: List[str] = None):
    """Stream response with typewriter effect."""
    words = response_text.split()
    current_text = ""
    
    for i, word in enumerate(words):
        current_text += word + " "
        
        # Format the response as a proper SSE message
        message = {
            "type": "stream",
            "content": current_text.strip()
        }
        yield f"data: {json.dumps(message)}\n\n"
        await asyncio.sleep(0.05)  # Add small delay for typewriter effect
    
    # Send next prompts if available
    if next_prompts:
        message = {
            "type": "next_prompts",
            "content": next_prompts
        }
        yield f"data: {json.dumps(message)}\n\n"
    
    # Send end message
    yield f"data: {json.dumps({'type': 'end'})}\n\n"

# Initialize FastAPI app
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
@limiter.limit("60/minute")
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, api_key: str = Depends(verify_api_key)) -> StreamingResponse:
    """Chat endpoint that streams responses with rate limiting and authentication."""
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        conversation_id = data.get("conversation_id", "default")

        if not message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )

        response, next_prompts = await get_chat_response(message, conversation_id)
        return StreamingResponse(
            stream_response(response, next_prompts),
            media_type="text/event-stream"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )
