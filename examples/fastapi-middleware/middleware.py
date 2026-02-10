"""
FastAPI middleware that automatically tokenizes PII in request bodies
and provides a detokenize helper on request.state.
"""

import json

from blindfold import Blindfold
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class BlindfoldMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str, policy: str = "basic", text_field: str = "text"):
        super().__init__(app)
        self.blindfold = Blindfold(api_key=api_key)
        self.policy = policy
        self.text_field = text_field

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            body = await request.body()
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = None

            if data and isinstance(data.get(self.text_field), str):
                text = data[self.text_field]
                result = self.blindfold.tokenize(text, policy=self.policy)

                # Store context for downstream route handlers
                request.state.blindfold = {
                    "original_text": text,
                    "tokenized_text": result.text,
                    "mapping": result.mapping,
                }

                # Replace body with tokenized version
                data[self.text_field] = result.text
                request._body = json.dumps(data).encode()

        response = await call_next(request)
        return response
