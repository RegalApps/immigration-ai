# -*- coding: utf-8 -*-

import os
import sys
from typing import List, Dict, Optional, Any, Union, Tuple
from functools import lru_cache
import json
import time
import signal
import logging
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta

# OpenAI imports
import openai
from openai.embeddings_utils import get_embedding

# FastAPI imports
from fastapi import FastAPI, Request, HTTPException, Depends, Security
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

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

# Set up API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="Could not validate API key"
        )
    return api_key

# PDF generation imports
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

async def stream_response(response_text: str):
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
    
    # Send end message
    yield f"data: {json.dumps({'type': 'end'})}\n\n"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
MAX_CACHE_SIZE = 1000  # Maximum number of items in each cache
CACHE_TTL = 3600  # Default cache TTL in seconds
RESPONSE_CACHE_TTL = 300  # Response cache TTL in seconds
CHUNK_SIZE = 1500  # Default chunk size for text processing

@dataclass
class ConversationState:
    """Tracks the state of a conversation with a user."""
    current_question_group: Optional[str] = None
    questions_asked: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    answers_received: Dict[str, Dict[str, str]] = field(default_factory=lambda: defaultdict(dict))

    def add_answer(self, group: str, question: str, answer: str) -> None:
        """Add an answer to the conversation state."""
        self.answers_received[group][question] = answer
        if question not in self.questions_asked[group]:
            self.questions_asked[group].append(question)

    def get_next_question(self, group: str, questions: List[str]) -> Optional[str]:
        """Get the next unanswered question from a group."""
        for question in questions:
            if question not in self.questions_asked[group]:
                return question
        return None

@dataclass
class Cache:
    """Thread-safe cache with size limit and TTL."""
    ttl_seconds: int
    max_size: int = MAX_CACHE_SIZE
    _cache: Dict[str, Tuple[Any, float]] = field(default_factory=dict)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache if it exists and hasn't expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp <= self.ttl_seconds:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set a value in cache with timestamp."""
        self._clear_expired()
        if len(self._cache) >= self.max_size:
            # Remove oldest item if cache is full
            oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
            del self._cache[oldest_key]
        self._cache[key] = (value, time.time())

    def _clear_expired(self) -> None:
        """Remove expired items from cache."""
        current_time = time.time()
        expired_keys = [
            k for k, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for k in expired_keys:
            del self._cache[k]

# Initialize caches with size limits
chunk_cache = Cache(ttl_seconds=CACHE_TTL)
embedding_cache = Cache(ttl_seconds=CACHE_TTL)
response_cache = Cache(ttl_seconds=RESPONSE_CACHE_TTL)

LEGAL_CONTEXT = """I am Natalie, an expert immigration advisor specializing in helping extraordinary individuals 
navigate their US immigration journey, particularly in the technology sector. I combine deep legal expertise 
with genuine care for each person's unique situation.

MY APPROACH:
- I am warm and empathetic, but always focused on achieving immigration success
- I provide clear, actionable guidance based on proven pathways
- I'm direct about requirements and potential challenges
- I help identify and strengthen qualifying evidence
- I keep conversations on track toward concrete immigration solutions

GUIDING PRINCIPLES:
- Every question gets a clear, actionable answer
- Empathy without compromising on requirements
- Honest assessment of chances and alternatives
- Strategic thinking about long-term immigration goals
- Proactive identification of potential issues

IMMIGRATION EXPERTISE (2025):
1. Strategic Assessment:
   - Quick evaluation of strongest visa pathways
   - Clear identification of evidence gaps
   - Realistic timeline expectations
   - Risk assessment and mitigation
   - Alternative pathway planning

2. Success Requirements:
   - Specific evidence thresholds
   - Documentation standards
   - Timeline considerations
   - Common pitfalls to avoid
   - Strategic strengthening of cases

CURRENT PRIORITIES:
- Focus on concrete evidence gathering
- Clear qualification assessment
- Strategic documentation planning
- Timeline management
- Risk mitigation strategies

When discussing cases:
1. Assess current qualifications
2. Identify evidence gaps
3. Provide specific action items
4. Set clear expectations
5. Address potential challenges
6. Outline next steps

COMMUNICATION STYLE:
- Warm but focused on solutions
- Clear and direct about requirements
- Supportive while maintaining standards
- Strategic in guidance
- Proactive about potential issues"""

SYSTEM_INSTRUCTION = """Act as Natalie, an immigration expert who combines warmth with unwavering focus on 
immigration success. While being empathetic and understanding, always guide the conversation toward concrete 
immigration solutions. Be direct about requirements and challenges, but deliver this information with care 
and support. Every response should move the candidate closer to their immigration goals through clear, 
actionable guidance."""

INITIAL_QUESTIONS = [
    "What is your current visa status and when does it expire?",
    "Could you tell me about your educational background - degrees, majors, and institutions?",
    "What's your current role and experience level in your field?",
    "Have you had any previous US visas?",
    "Are you currently employed? If so, what's your role and company?",
    "For founders: Could you tell me about your company's stage and your role in it?",
    "Do you have family members who would need visa consideration?",
    "Are there any immediate timeline concerns we should discuss?"
]

WELCOME_MESSAGE = """Hello fellow tech enthusiast! 👋 I'm Natalie, and I specialize in helping professionals like yourself navigate the US immigration system. With the recent focus on AI/ML pathways and emerging technologies, I'm particularly excited to explore your options together.

I understand immigration can feel overwhelming, but I'm here to make this journey clearer for you. Let's start by getting to know your situation better."""

FOLLOW_UP_QUESTIONS = {
    'recent_grad': [
        "Could you tell me more about your internship experience. What kind of projects did you work on?",
        "Did you participate in any research projects during your studies?",
        "Have you published any papers or contributed to open-source projects?",
        "Were you involved in any notable hackathons or coding competitions?",
        "Did you receive any academic awards or scholarships?",
        "What was your GPA, and did you have any specializations?",
        "Are you currently employed or have any job offers?",
        "Are you on F-1 status? If so, have you applied for OPT?",
        "Do you have any specific companies or roles in mind?",
        "What are your long-term career goals in the US?"
    ],
    'tech_professional': [
        "Could you share more about your specific achievements in your current role?",
        "Have you led any significant projects or innovations?",
        "Do you have any patents or publications?",
        "Have you received recognition from industry peers or media?",
        "What's your role in the tech community (speaking, mentoring, etc.)?",
        "Do you have any unique expertise or specializations?",
        "Are you currently working with a US company?",
        "What's your current visa status and timeline?",
        "Have you considered multiple visa pathways?",
        "What are your long-term goals in the US tech industry?"
    ],
    'startup_founder': [
        "Could you tell me about your company's current stage and funding?",
        "What's your role and ownership percentage?",
        "Have you received any notable investments or recognition?",
        "Do you have any patents or innovative technologies?",
        "What's your company's potential impact on the US market?",
        "How many employees do you have or plan to hire?",
        "What's your timeline for US market entry?",
        "Have you established any US business relationships?",
        "What's your current location and business structure?",
        "What are your growth projections for the next 2-3 years?"
    ]
}

SOLUTION_FRAMEWORKS = {
    'recent_grad': {
        'immediate_options': [
            "OPT (12 months + 24-month STEM extension)",
            "H-1B through employer sponsorship",
            "Cap-exempt H-1B opportunities",
            "J-1 training or research programs",
            "O-1A preparation strategy"
        ],
        'medium_term': [
            "Advanced degree pursuit (Master's/PhD)",
            "Building specialized expertise",
            "Research and publications",
            "Industry recognition development",
            "Professional network building"
        ],
        'long_term': [
            "EB-1A qualification building",
            "EB-2 NIW pathway",
            "Employer-sponsored options",
            "Entrepreneurship pathways",
            "Dual intent visa strategies"
        ]
    }
}

# Personality traits for different response types
PERSONALITY_TRAITS = {
    "greeting": [
        "I'm Natalie, your dedicated immigration specialist. How may I assist you with your immigration journey today?",
        "Welcome, I'm Natalie, your immigration advisor. I'm here to guide you through the immigration process. What can I help you with?",
        "Greetings, I'm Natalie, your immigration partner. I specialize in helping professionals navigate their immigration path. How can I assist you?"
    ],
    "understanding": [
        "I understand your immigration concerns. Let me help you navigate this.",
        "Based on what you've shared, I can guide you through the next steps.",
        "I see your situation clearly. Let me provide you with specific guidance."
    ],
    "clarification": [
        "To provide you with the most accurate guidance, could you please clarify something for me?",
        "Let me ensure I have all the details correct. Could you confirm something?",
        "For the most precise advice, I need to understand one more detail."
    ],
    "empathy": [
        "I recognize this process can be complex. Rest assured, I'm here to guide you through each step.",
        "Immigration procedures can be challenging, but together we'll navigate them effectively.",
        "I understand your concerns. Let's address them systematically to ensure the best outcome."
    ],
    "action": [
        "Based on your situation, here are the specific steps we should take:",
        "Let me outline the precise actions needed for your case:",
        "Here's what I recommend as your next steps:"
    ],
    "follow_up": [
        "Is there anything specific about these steps you'd like me to explain further?",
        "Would you like me to provide more details about any of these points?",
        "Do you have any questions about the process I've outlined?"
    ],
    "closing": [
        "I'm here to help if you have any more questions. Feel free to ask!",
        "Don't hesitate to ask if you need clarification on anything I've explained.",
        "I'm always happy to provide more detailed guidance if needed. Just let me know!"
    ]
}

# Conversation starters based on client type
CONVERSATION_STARTERS = {
    "tech_worker": [
        "Welcome. As your immigration specialist, I'm here to help with your tech industry immigration needs. What specific visa or immigration matter can I assist you with?",
        "I specialize in tech industry immigration. Whether it's H-1B, O-1, or other visa categories, I'm here to guide you. What's your current situation?"
    ],
    "student": [
        "Welcome. I'm here to assist with your student immigration matters. Are you interested in F-1 visas, OPT, or other student immigration options?",
        "As your immigration advisor, I can help navigate student visa processes. What specific aspect of student immigration would you like to discuss?"
    ],
    "default": [
        "Welcome. I'm Natalie, your dedicated immigration specialist. I'm here to provide expert guidance on your immigration journey. What specific assistance do you need?",
        "I'm Natalie, your immigration advisor. I'm here to help you navigate the immigration process effectively. What immigration matters would you like to discuss?"
    ]
}

# Global conversation state
conversations: Dict[str, ConversationState] = {}

QUESTION_GROUPS = {
    'recent_grad': {
        'background': [
            "I'd love to hear more about your internship experience. What kind of projects did you work on?",
            "Did you have the chance to do any research projects during your studies?",
            "Were you involved in any coding competitions or hackathons?"
        ],
        'achievements': [
            "Have you published any papers or made contributions to open-source projects?",
            "Did you receive any academic awards or scholarships?",
            "What was your favorite project or accomplishment during your studies?"
        ],
        'goals': [
            "What kind of work are you most passionate about in computer science?",
            "Do you have any specific companies or roles you're interested in?",
            "Where do you see yourself in the next few years?"
        ]
    },
    'tech_professional': {
        'experience': [
            "Could you tell me more about your current role and the impact you've made?",
            "What kind of innovative projects have you led or contributed to?",
            "Have you mentored others or led any teams?"
        ],
        'recognition': [
            "Have you received any recognition or awards in your field?",
            "Could you share any notable achievements or breakthroughs?",
            "Have you spoken at any conferences or published any papers?"
        ]
    }
}

class ImmigrationCase:
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.details = {}
        self.documents = []
        self.timeline = []
        
    def update_details(self, details: dict):
        self.details.update(details)
        
    def add_document(self, document: str):
        self.documents.append(document)
        
    def add_timeline_event(self, event: dict):
        self.timeline.append(event)

# Active cases storage
active_cases = {}

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

async def get_relevant_context(query: str, chunks: List[str] = None) -> str:
    """Get relevant context for a query using semantic search."""
    try:
        # Get chunks if not provided
        if chunks is None:
            with open("immigration.txt", "r", encoding="utf-8") as f:
                text = f.read()
            chunks = await get_cached_chunks(text)
        
        # Get query embedding
        query_response = await openai.Embeddings.create(
            model="text-embedding-ada-002",
            input=query,
            encoding_format="float"
        )
        query_embedding = query_response.data[0].embedding
        
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
        
        return "\n\n".join(relevant_chunks)
    except Exception as e:
        logger.error(f"Error getting relevant context: {e}")
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
    cache_key = f"{hash(text)}_{chunk_size}"
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
        chunk_key = hash(chunk)
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
            response = await openai.Embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            
            # Cache and insert embeddings
            for j, embedding in enumerate(response.data):
                chunk = to_process[i + j]
                embedding_cache.set(hash(chunk), embedding.embedding)
                embeddings.insert(indices[i + j], embedding.embedding)
    
    return embeddings

async def get_chat_response(message: str) -> str:
    try:
        # Get relevant context from immigration text
        context = await get_relevant_context(message)
        
        # Create messages for the chat
        messages = [
            {"role": "system", "content": LEGAL_CONTEXT},
            {"role": "user", "content": f"Based on this context:\n\n{context}\n\nQuestion: {message}"}
        ]
        
        # Get OpenAI response with function calling
        completion = await openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=get_functions_for_openai(),
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        # Extract the response
        response_message = completion.choices[0].message

        # Check if the model wants to call a function
        if response_message.function_call is not None:
            # Get function details
            function_name = response_message.function_call.name
            try:
                function_args = json.loads(response_message.function_call.arguments)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse function arguments: {response_message.function_call.arguments}")
                return "I apologize, but I encountered an error processing your request. Please try again."
            
            # Call the function
            function_response = await handle_function_call(function_name, function_args)
            
            # Add function response to conversation
            messages.append(response_message)
            messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_response)
            })
            
            # Get final response
            final_completion = await openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return final_completion.choices[0].message.content
        
        return response_message.content

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

@app.get("/")
@limiter.limit("60/minute")
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
@limiter.limit("30/minute")
async def chat(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> StreamingResponse:
    """Chat endpoint that streams responses with rate limiting and authentication."""
    try:
        data = await request.json()
        query = data.get("message", "")
        conversation_id = data.get("conversation_id", "default")

        # Initialize conversation state if needed
        if conversation_id not in conversations:
            conversations[conversation_id] = ConversationState()

        # Generate response
        response = await get_chat_response(query)
        
        # Stream the response
        return StreamingResponse(
            stream_response(response),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/download/{filename}")
@limiter.limit("10/minute")
async def download_pdf(
    filename: str,
    api_key: str = Depends(verify_api_key)
):
    """Serve generated PDF files with rate limiting and authentication."""
    file_path = f"generated_pdfs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename=filename
        )
    return {"error": "File not found"}

def find_available_port(start_port: int = 8000, max_tries: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_tries}")

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination request
    
    # Find available port
    try:
        port = find_available_port()
        print(f"\nStarting server on port {port}")
        
        # Configure and start server
        server_config = uvicorn.Config(app, host="0.0.0.0", port=port)
        server = uvicorn.Server(server_config)
        config['server'] = server
        
        server.run()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Shutting down...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("Server shutdown complete.")

IMMIGRATION_SOURCES = {
    "uscis_policy": {
        "name": "USCIS Policy Manual",
        "url": "https://www.uscis.gov/policy-manual",
        "citation_format": "USCIS Policy Manual, Volume {volume}, Part {part}, Chapter {chapter}"
    },
    "ina": {
        "name": "Immigration and Nationality Act",
        "url": "https://www.uscis.gov/laws-and-policy/legislation/immigration-and-nationality-act",
        "citation_format": "INA § {section}"
    },
    "cfr": {
        "name": "Code of Federal Regulations",
        "url": "https://www.ecfr.gov/current/title-8",
        "citation_format": "8 CFR § {section}"
    }
}

def get_personality_response(category: str) -> str:
    """Get a random personality response from the specified category."""
    return random.choice(PERSONALITY_TRAITS[category])

def get_conversation_starter(client_type: str) -> str:
    """Get a random conversation starter for the client type."""
    return random.choice(CONVERSATION_STARTERS.get(client_type, CONVERSATION_STARTERS["default"]))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_functions_for_openai():
    """Convert AVAILABLE_FUNCTIONS to OpenAI's format."""
    functions = []
    for name, details in AVAILABLE_FUNCTIONS.items():
        function = {
            "name": name,
            "description": details["description"],
            "parameters": details["parameters"]
        }
        functions.append(function)
    return functions
