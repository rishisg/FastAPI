################################################################################
# FastAPI Service for Summarization:
# 
# This FastAPI application provides an API for Summarization.
# It interacts with multiple AI providers like Gemini and Mistral to perform Summarization.
# 
# The application requires Basic Authentication, and credentials are loaded from the `.env` file.
# 
# Supported AI Providers:
# - Gemini (via Google Generative AI)
# - Mistral (via Mistral API)
#
# Endpoints:
# - POST /summarize: Takes input text and performs the operation based on the selected provider.
#
# Example:
# {
#  "text": "This is a long paragraph of text that needs to be summarized. It could be a lengthy article, blog post, or anything want to short.",
#  "provider": "gemini"
# }
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
import google.generativeai as genai
from passlib.context import CryptContext

# Load environment variables from .env file
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

# SummarizationRequest Pydantic model with default provider
class SummarizationRequest(BaseModel):
    text: str
    provider: str = Field(default="gemini", example="gemini")

# Enum for selecting providers (Gemini and Mistral for now)
class Provider(str, Enum):
    gemini = "gemini"
    mistral = "mistral"

# Function to summarize with Gemini
def summarize_with_gemini(text: str):
    gemini_model = None
    if os.getenv("GEMINI_API_KEY"):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            response = gemini_model.generate_content(f"Summarize the following text: {text}")
            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Gemini model not initialized")

# Function to summarize with Mistral
def summarize_with_mistral(text: str):
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
            "messages": [{"role": "user", "content": f"Summarize the following text: {text}"}]
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

# POST endpoint for summarizing text
@app.post("/summarize")
async def summarize(request: SummarizationRequest, username: str = Depends(verify_credentials)):
    text = request.text
    provider = request.provider  # This will default to "gemini" if not provided

    try:
        if provider == "gemini":
            summary = summarize_with_gemini(text)
        elif provider == "mistral":
            summary = summarize_with_mistral(text)
        else:
            raise HTTPException(status_code=400, detail="Invalid provider specified")

        return {
            "text": text,
            "provider": provider,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI server runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
