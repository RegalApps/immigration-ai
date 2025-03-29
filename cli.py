import asyncio
import sys
import random
from immigration_chatbot import (
    get_chat_response, chunk_text, get_relevant_context,
    WELCOME_MESSAGE, get_personality_response, PERSONALITY_TRAITS
)

async def main():
    # Check if question is provided as argument
    if len(sys.argv) < 2:
        print("Usage: python3 cli.py \"your question here\"")
        return
        
    query = " ".join(sys.argv[1:])
    
    print("\n=== Immigration Law Assistant ===")
    print(WELCOME_MESSAGE)
    print("=" * 35)
    
    # Load immigration text
    try:
        with open("immigration.txt", "r", encoding="utf-8") as f:
            immigration_text = f.read()
    except FileNotFoundError:
        print("\nError: immigration.txt not found. Please create this file with your immigration text.")
        return

    # Chunk the text
    chunks = chunk_text(immigration_text)
    print(f"\nText split into {len(chunks)} chunks")
    
    try:
        # Get relevant context and generate response
        relevant_context = await get_relevant_context(query, chunks)
        response = await get_chat_response(query)
        
        print("\nQuestion:", query)
        print("\nResponse:", response)
        print("-" * 50)
        
        # Add contextual follow-up suggestions based on the query content
        if "h-1b" in query.lower():
            print("\nSome follow-up questions you might consider:")
            print("1. What is the H-1B cap and lottery process?")
            print("2. Can I work for multiple employers on an H-1B?")
            print("3. What happens to my H-1B status if I change jobs?")
        elif "visa" in query.lower():
            print("\nSome follow-up questions you might consider:")
            print("1. What are the processing times for this visa category?")
            print("2. What are the key differences between this and other visa options?")
            print("3. Can my family members accompany me on this visa?")
        
        # Add a personality-driven follow-up message
        print("\n" + random.choice(PERSONALITY_TRAITS["follow_up"]))
        print(get_personality_response("closing"))
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
