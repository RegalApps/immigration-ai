import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv(os.path.join(SCRIPT_DIR, '.env'))

class LegalTextProcessor:
    def __init__(self):
        self.chunk_size = 500  # Smaller chunks for better retrieval
        self.chunk_overlap = 50  # Reduced overlap
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.vector_store_path = os.path.join(SCRIPT_DIR, "vector_store")
        
        # Initialize text splitter with legal-specific settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # Legal text specific separators
            is_separator_regex=False
        )
        
        # Initialize embeddings with error handling
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise

    def load_and_split_text(self, file_path: str) -> List[str]:
        """Load and split text with improved chunking for legal documents."""
        try:
            abs_file_path = os.path.join(SCRIPT_DIR, file_path)
            chunks = []
            
            # Process file in chunks to handle large files
            with open(abs_file_path, 'r') as file:
                current_text = []
                current_size = 0
                
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                        
                    line_size = len(line)
                    if current_size + line_size > self.chunk_size:
                        # Process current chunk
                        if current_text:
                            text = " ".join(current_text)
                            sub_chunks = self.text_splitter.split_text(text)
                            chunks.extend(sub_chunks)
                        current_text = [line]
                        current_size = line_size
                    else:
                        current_text.append(line)
                        current_size += line_size
                
                # Process final chunk
                if current_text:
                    text = " ".join(current_text)
                    sub_chunks = self.text_splitter.split_text(text)
                    chunks.extend(sub_chunks)
            
            # Validate chunks
            valid_chunks = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Minimum chunk size
                    valid_chunks.append(chunk)
            
            logger.info(f"Successfully split text into {len(valid_chunks)} chunks")
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            raise

    def create_vector_store(self, text_chunks: List[str]) -> FAISS:
        """Create FAISS vector store with improved embedding and caching."""
        try:
            # Check for existing vector store
            if os.path.exists(self.vector_store_path):
                logger.info("Loading existing vector store...")
                return FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Safe since we created the index ourselves
                )
            
            logger.info("Creating new vector store...")
            vector_store = FAISS.from_texts(text_chunks, self.embeddings)
            
            # Save vector store for future use
            vector_store.save_local(self.vector_store_path)
            logger.info("Vector store created and saved successfully")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

def get_yes_no_input(prompt):
    """Get a yes/no input from the user, handling various forms of yes/no responses."""
    while True:
        response = input(prompt).strip().lower()
        if response in ['yes', 'y', 'ye', 'yeah', 'correct', 'true', '1']:
            return "yes"
        elif response in ['no', 'n', 'nah', 'incorrect', 'false', '0']:
            return "no"
        else:
            print("Please answer with 'yes' or 'no'.")

def get_valid_input(prompt, allow_empty=False, max_length=None):
    """Get input from the user and validate it's not empty unless allowed."""
    while True:
        try:
            response = input(prompt).strip()
            if not response and not allow_empty:
                print("Please provide a valid response.")
                continue
            if max_length and len(response) > max_length:
                print(f"Response too long. Please keep it under {max_length} characters.")
                continue
            return response
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            print("Please try again.")

def get_user_context():
    """Gather relevant context from the user."""
    try:
        print("\nTo provide you with the most accurate immigration law information, please share some context:")
        nationality = get_valid_input("What is your nationality? (e.g., Canadian, Indian, Chinese): ", max_length=50)
        current_status = get_valid_input("What is your current immigration status (if any)? ", allow_empty=True, max_length=100)
        field = get_valid_input("What is your field of work/expertise? ", max_length=100)
        experience = get_valid_input("How many years of experience do you have in this field? ", max_length=10)
        
        # Add funding-related questions
        has_funding = get_yes_no_input("Have you raised funding capital? (yes/no): ")
        funding_amount = "0"
        if has_funding == "yes":
            funding_amount = get_valid_input("How much funding have you raised? (in USD): ", max_length=20)
            try:
                # Convert to float and format with commas
                funding_amount = "{:,.2f}".format(float(funding_amount.replace(",", "")))
            except ValueError:
                print("Warning: Could not parse funding amount. Using raw input.")
        
        # Add professional recommendations questions
        has_recommendations = get_yes_no_input("Can you get signed recommendation letters from extraordinary American technology professionals? (yes/no): ")
        recommendation_details = ""
        if has_recommendations == "yes":
            recommendation_details = get_valid_input("Please describe the professionals who can provide recommendations (e.g., YC partners, industry leaders, etc.): ", max_length=500)
        
        # Add publications and media coverage questions
        has_publications = get_yes_no_input("Do you have publications in scientific/technical journals? (yes/no): ")
        publication_details = ""
        if has_publications == "yes":
            publication_details = get_valid_input("Please describe your publications (e.g., journal names, impact factors, citations): ", max_length=500)
        
        has_media_coverage = get_yes_no_input("Do you have media coverage (e.g., Forbes, TechCrunch, major tech publications)? (yes/no): ")
        media_details = ""
        if has_media_coverage == "yes":
            media_details = get_valid_input("Please describe your media coverage (e.g., publication names, article topics): ", max_length=500)
        
        has_competition_experience = get_yes_no_input("Have you judged or won American engineering competitions or hackathons? (yes/no): ")
        competition_details = ""
        if has_competition_experience == "yes":
            competition_details = get_valid_input("Please describe your competition experience (e.g., competition names, roles, outcomes): ", max_length=500)
        
        # Store context as a dictionary with keys matching prompt template variables
        context_dict = {
            "nationality": nationality,
            "current_status": current_status,
            "field": field,
            "experience": experience,
            "funding_status": "Yes" if has_funding == "yes" else "No",
            "funding_amount": funding_amount,
            "recommendations": "Yes" if has_recommendations == "yes" else "No",
            "recommendation_details": recommendation_details if recommendation_details else "N/A",
            "publications": "Yes" if has_publications == "yes" else "No",
            "publication_details": publication_details if publication_details else "N/A",
            "media": "Yes" if has_media_coverage == "yes" else "No",
            "media_details": media_details if media_details else "N/A",
            "competition": "Yes" if has_competition_experience == "yes" else "No",
            "competition_details": competition_details if competition_details else "N/A"
        }
        
        # Create a formatted string for display
        context_string = """User Context:
Nationality: {}
Current Status: {}
Field: {}
Years of Experience: {}
Funding Status: {}
Funding Amount: ${}
Professional Recommendations: {}
Recommendation Details: {}
Scientific Publications: {}
Publication Details: {}
Media Coverage: {}
Media Details: {}
Competition Experience: {}
Competition Details: {}""".format(
            nationality, 
            current_status, 
            field, 
            experience,
            "Yes" if has_funding == "yes" else "No",
            funding_amount,
            "Yes" if has_recommendations == "yes" else "No",
            recommendation_details if recommendation_details else "N/A",
            "Yes" if has_publications == "yes" else "No",
            publication_details if publication_details else "N/A",
            "Yes" if has_media_coverage == "yes" else "No",
            media_details if media_details else "N/A",
            "Yes" if has_competition_experience == "yes" else "No",
            competition_details if competition_details else "N/A"
        )
        
        return context_dict, context_string
    except KeyboardInterrupt:
        print("\nExiting...")
        raise
    except Exception as e:
        print("Error gathering user context: {}".format(str(e)))
        raise

def setup_qa_chain(vector_store: FAISS, user_context_dict: Dict[str, str], text_processor: LegalTextProcessor):
    """Set up the QA chain with improved context handling."""
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # Create a custom prompt template
        prompt_template = """You are a legal expert specializing in U.S. immigration law with extensive experience in technology sector visas. Analyze the following information and provide a detailed response.

User Background Information:
Nationality: {nationality}
Current Status: {current_status}
Field: {field}
Years of Experience: {experience}
Funding Status: {funding_status}
Funding Amount: ${funding_amount}
Professional Recommendations: {recommendations}
Recommendation Details: {recommendation_details}
Scientific Publications: {publications}
Publication Details: {publication_details}
Media Coverage: {media}
Media Details: {media_details}
Competition Experience: {competition}
Competition Details: {competition_details}

Relevant Legal Context:
{context}

Question: {question}

Instructions:
1. Based on the user's background and legal context, provide a detailed answer tailored to their situation
2. Include specific case citations with years when referencing legal precedents
3. When relevant, reference recent success cases from similar technology professionals
4. Highlight specific evidence categories and documentation strategies that would be most relevant for their profile
5. Provide actionable recommendations based on their background
6. If certain information is missing or unclear, mention what additional details would be helpful
7. If you don't know the answer, say so directly - do not speculate
8. For funding-related questions:
   - If the user has raised over $500,000, emphasize this as a strong indicator of extraordinary ability
   - Compare funding amounts to industry standards (e.g., YC funding rounds)
   - Highlight how funding demonstrates business sustainability and job creation potential
9. For professional recommendations:
   - Emphasize the value of recommendation letters from prominent figures in the tech industry
   - Reference successful cases where YC partners and industry leaders provided support
   - Highlight how such recommendations demonstrate recognition by peers
10. For publications and technical contributions:
    - Highlight the significance of scientific/technical publications
    - Emphasize impact factors and citation counts when available
    - Reference how publications demonstrate expertise and influence in the field
11. For media coverage and public recognition:
    - Emphasize coverage in major tech publications (Forbes, TechCrunch)
    - Highlight how media coverage demonstrates industry impact
    - Reference how public recognition supports extraordinary ability claims
12. For competition experience:
    - Emphasize judging roles in American engineering competitions
    - Highlight winning or significant participation in hackathons
    - Reference how competition experience demonstrates peer recognition
13. Consider current political climate:
    - Acknowledge any specific challenges or considerations for the applicant's nationality
    - Provide balanced, factual information about current immigration policies
    - Focus on evidence-based qualifications rather than political factors
    - Include relevant recent precedents or policy changes

Your response should be structured as follows:
1. Direct answer to the question
2. Relevant case examples and precedents
3. Specific evidence categories that apply to their situation
4. Documentation strategies and recommendations
5. Next steps or additional information needed

Your response:"""

        # Create the chain with the correct prompt format
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.8  # Only return highly relevant chunks
                }
            ),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt_template,
                    input_variables=[
                        "context", "question",
                        "nationality", "current_status", "field", "experience",
                        "funding_status", "funding_amount",
                        "recommendations", "recommendation_details",
                        "publications", "publication_details",
                        "media", "media_details",
                        "competition", "competition_details"
                    ]
                )
            }
        )
        
        def qa_chain_with_context(query):
            try:
                # Create a copy of the context dictionary to avoid modifying the original
                formatted_query = user_context_dict.copy()
                formatted_query["question"] = query  # Changed from "query" to "question"
                
                # Get relevant chunks for better context
                docs = vector_store.similarity_search_with_score(query, k=5)
                context_chunks = []
                for doc, score in sorted(docs, key=lambda x: x[1]):
                    if score < 0.8:  # Only include highly relevant chunks
                        context_chunks.append(doc.page_content)
                
                # Combine context chunks
                formatted_query["context"] = "\n\n".join(context_chunks)
                
                response = chain.invoke(formatted_query)
                return {"result": response.get("result", "")}
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                raise
        
        return qa_chain_with_context
    except Exception as e:
        logger.error(f"Error setting up QA chain: {str(e)}")
        raise

def test_chatbot(qa_chain):
    """Test the chatbot with fabricated queries."""
    test_queries = [
        "Am I eligible for an O-1 visa based on my background?",
        "What specific evidence should I gather for my O-1 application?",
        "How does my funding amount compare to industry standards?",
        "What are the key success factors for technology professionals in O-1 cases?",
        "How can I strengthen my application with my current qualifications?",
        "What are the common pitfalls to avoid in O-1 applications?",
        "How does my nationality affect my O-1 application?",
        "What recent success cases are similar to my profile?",
        "How important are recommendation letters in my case?",
        "What documentation strategies would work best for my background?"
    ]
    
    print("\nStarting automated test with fabricated queries...")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\nTest Query {i}: {query}")
            print("-" * 40)
            response = qa_chain(query)
            if 'result' in response:
                print("\nResponse:")
                print(response['result'])
            else:
                print("\nError: Unexpected response format")
            print("=" * 80)
        except Exception as e:
            print(f"\nError processing query {i}: {str(e)}")
            print("=" * 80)
    
    print("\nTest completed!")

def main():
    try:
        # Initialize text processor
        text_processor = LegalTextProcessor()
        
        # Load and process the text
        text_chunks = text_processor.load_and_split_text("case_law.txt")
        
        # Create vector store
        vector_store = text_processor.create_vector_store(text_chunks)
        
        # Get user context
        user_context_dict, _ = get_user_context()
        
        # Set up QA chain
        qa_chain = setup_qa_chain(vector_store, user_context_dict, text_processor)
        
        # Run test queries
        test_chatbot(qa_chain)
        
        # Ask if user wants to continue with manual queries
        while True:
            try:
                continue_manual = input("\nWould you like to continue with manual queries? (yes/no): ").strip().lower()
                if continue_manual not in ['yes', 'y']:
                    print("\nThank you for using the legal chatbot. Goodbye!")
                    break
                
                # Get user query
                query = input("\nWhat is your immigration law question? (type 'exit' to quit): ").strip()
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                if query.lower() == 'exit':
                    print("\nThank you for using the legal chatbot. Goodbye!")
                    break
                
                # Get response
                response = qa_chain(query)
                if 'result' in response:
                    print("\nAnswer:")
                    print(response['result'])
                    print("\n" + "-" * 80)
                else:
                    print("\nError: Unexpected response format")
                
            except KeyboardInterrupt:
                print("\nExiting the chatbot...")
                break
            except Exception as e:
                print("\nAn error occurred while processing your question: {}".format(str(e)))
                print("Please try asking your question again.")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print("Fatal error: {}".format(str(e)))
        print("The chatbot could not continue due to an error.")

if __name__ == "__main__":
    main() 