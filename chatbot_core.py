#!/usr/bin/env python3
import os
from openai import AsyncOpenAI
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
import json
import asyncio

# Load environment variables
load_dotenv()

# Configure OpenAI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cache for embeddings
EMBEDDING_CACHE: Dict[str, List[float]] = {}
CACHE_FILE = "embedding_cache.json"

# Load cache if exists
try:
    with open(CACHE_FILE, 'r') as f:
        EMBEDDING_CACHE = json.load(f)
except FileNotFoundError:
    pass

async def get_embedding(text: str, model="text-embedding-ada-002") -> List[float]:
    """Get embedding for a text using OpenAI's API with caching."""
    text = text.replace("\n", " ")
    
    # Check cache
    if text in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text]
    
    # Get new embedding
    response = await client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    
    # Cache result
    EMBEDDING_CACHE[text] = embedding
    
    # Save cache periodically (every 10 new embeddings)
    if len(EMBEDDING_CACHE) % 10 == 0:
        with open(CACHE_FILE, 'w') as f:
            json.dump(EMBEDDING_CACHE, f)
    
    return embedding

async def get_embeddings_batch(texts: List[str], model="text-embedding-ada-002") -> List[List[float]]:
    """Get embeddings for multiple texts in parallel."""
    tasks = []
    for text in texts:
        if text not in EMBEDDING_CACHE:
            tasks.append(get_embedding(text, model))
        else:
            tasks.append(None)
    
    # Run non-cached embeddings in parallel
    if tasks:
        results = await asyncio.gather(*[t for t in tasks if t is not None])
    
    # Combine cached and new results
    embeddings = []
    result_index = 0
    for i, text in enumerate(texts):
        if text in EMBEDDING_CACHE:
            embeddings.append(EMBEDDING_CACHE[text])
        else:
            embeddings.append(results[result_index])
            result_index += 1
    
    return embeddings

async def get_relevant_context(query: str, chunks: List[str] = None) -> str:
    """Get relevant context for a query using semantic search."""
    try:
        # Load and chunk text if not provided
        if not chunks:
            with open("immigration.txt", "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_text(text)

        # Get query and chunk embeddings in parallel
        query_embedding = await get_embedding(query)
        chunk_embeddings = await get_embeddings_batch(chunks)

        # Calculate similarities
        similarities = []
        for emb in chunk_embeddings:
            similarity = np.dot(query_embedding, emb)
            similarities.append(similarity)

        # Get most relevant chunks
        top_k = 3
        top_indices = np.argsort(similarities)[-top_k:]
        relevant_chunks = [chunks[i] for i in top_indices]

        return "\n".join(relevant_chunks)

    except Exception as e:
        print(f"Error getting context: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

async def get_chat_response(message: str) -> str:
    """Get a response from the chatbot."""
    try:
        # Get relevant context
        context = await get_relevant_context(message)

        # Create messages for the chat
        messages = [
            {"role": "system", "content": "You are a helpful immigration lawyer with 10 years of experience, especially in tech industry immigration. Provide accurate, clear advice based on the context provided."},
            {"role": "user", "content": f"Based on this context:\n\n{context}\n\nQuestion: {message}"}
        ]

        # Get completion from OpenAI
        completion = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"
