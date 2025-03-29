#!/usr/bin/env python3
import sys
import asyncio
from chatbot_core import get_chat_response

async def main():
    """Main CLI function."""
    if len(sys.argv) < 2:
        print("Usage: python cli.py 'your question here'")
        return

    query = sys.argv[1]
    try:
        response = await get_chat_response(query)
        print("\nResponse:")
        print("-" * 80)
        print(response)
        print("-" * 80)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
