################################################################################
# FastAPI Service for Translation:
# 
# This FastAPI application provides an API for Translation.
# It interacts with multiple AI providers like Gemini and Mistral to perform Translation.
# 
# The application requires Basic Authentication, and credentials are loaded from the `.env` file.
# 
# Supported AI Providers:
# - Gemini (via Google Generative AI)
# - Mistral (via Mistral API)
#
# Endpoints:
# - POST /translate: Takes input text and performs the operation based on the selected provider.
#
# Example:
#  {
#    "text": "Hello, how are you?",
#    "source_language": "en",
#    "target_language": "fr",
#    "provider": "gemini"
#  }
#
# Requirements:
# - FastAPI
# - Uvicorn
# - Requests
# - Google Generative AI
# - Mistral API
# - Postman Collection
# - Curl
#- VS Code IDE
#
# Make sure to configure the `.env` file with the required credentials and API keys for the respective AI providers. See the details in README.md.
####################################################################

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from enum import Enum
import openai
import requests
import os
from dotenv import load_dotenv
from passlib.context import CryptContext
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize HTTPBasic security
security = HTTPBasic()

# Initialize password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Retrieve credentials from environment variables
API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")

# Hash the password from .env for comparison
hashed_password = pwd_context.hash(API_PASSWORD)

# Dependency to verify credentials
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != API_USERNAME or not pwd_context.verify(credentials.password, hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# TranslationRequest Pydantic model
class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str
    provider: str = "gemini"  # Default to gemini

# Enum for selecting providers
class Provider(str, Enum):
    gemini = "gemini"
    mistral = "mistral"

# Initialize model clients for Gemini
gemini_model = None

# Configure Gemini
try:
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 64,
                "max_output_tokens": 8192,
            }
        )
except Exception as e:
    print(f"Gemini initialization error: {e}")

# Function to translate with Gemini
def translate_with_gemini(text: str, source_language : str, target_language: str):
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini model not initialized")
    try:
        prompt = f"""Translate the following text to {target_language}: {text}"""
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

# Function to translate with Mistral
def translate_with_mistral(text: str, source_language: str, target_language: str):
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise HTTPException(status_code=500, detail="Mistral API key not configured")
    
    try:
        headers = {
            "Authorization": f"Bearer {mistral_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-tiny",
            "messages": [{"role": "user", "content": f"Translate the following text to {target_language}: {text}"}]
        }
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mistral error: {str(e)}")

# POST endpoint for translation
@app.post("/translate")
async def translate(request: TranslationRequest, username: str = Depends(verify_credentials)):
    text = request.text
    source_language = request.source_language
    target_language = request.target_language
    provider = request.provider  # This will default to "gemini" if not provided

    try:
        if provider == "gemini":
            translated_text = translate_with_gemini(text, source_language, target_language)
        elif provider == "mistral":
            translated_text = translate_with_mistral(text, source_language, target_language)
        else:
            raise HTTPException(status_code=400, detail="Invalid provider specified")

        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language,
            "provider": provider
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI server runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
