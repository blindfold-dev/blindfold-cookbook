"""
FastAPI server with Blindfold PII protection middleware.

POST /chat sends user messages to OpenAI with PII automatically
tokenized and detokenized.
"""

import os

from blindfold import Blindfold
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from openai import OpenAI
from pydantic import BaseModel

from middleware import BlindfoldMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    BlindfoldMiddleware,
    api_key=os.environ.get("BLINDFOLD_API_KEY"),
    policy="basic",
    text_field="message",
)

blindfold = Blindfold(api_key=os.environ.get("BLINDFOLD_API_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    tokenized_message = body.message  # Already tokenized by middleware
    print(f"Tokenized message: {tokenized_message}")

    # OpenAI only sees tokenized text — no real PII
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": tokenized_message},
        ],
    )
    ai_response = completion.choices[0].message.content

    # Detokenize the AI response to restore real values
    mapping = getattr(request.state, "blindfold", {}).get("mapping", {})
    if mapping:
        restored = blindfold.detokenize(ai_response, mapping)
        return {"response": restored.text}

    return {"response": ai_response}
