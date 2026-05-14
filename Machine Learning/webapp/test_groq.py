import asyncio
from app import call_ai

async def main():
    try:
        res = await call_ai("Hello", "You are an assistant", "groq")
        print("Success:", res)
    except Exception as e:
        print("Failed:", str(e))

asyncio.run(main())
