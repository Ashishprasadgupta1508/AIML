import os

from dotenv import load_dotenv
from pathlib import Path
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


class AIMLAPIKeyAuthentication(BaseAuthentication):

    def authenticate(self, request):

        expected_key = os.getenv("AIML_API_KEY")

        if not expected_key:
            raise AuthenticationFailed(
                "AIML_API_KEY is not configured."
            )

        provided_key = request.headers.get("X-AI-API-KEY")

        if not provided_key:
            raise AuthenticationFailed(
                "AI API key is required."
            )

        if provided_key != expected_key:
            raise AuthenticationFailed(
                "Invalid AI API key."
            )

        return (None, None)