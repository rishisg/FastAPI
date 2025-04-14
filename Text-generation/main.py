from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import google.generativeai as genai
from openai import OpenAI
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize model variables
gemini_model = None
openai_client = None

# Configure providers
if os.getenv("GEMINI_API_KEY"):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        print(f"Error initializing Gemini: {str(e)}")

if os.getenv("OPENAI_API_KEY"):
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"Error initializing OpenAI: {str(e)}")

# Provider enum for selection
class Provider(str, Enum):
    gemini = "gemini"
    openai = "openai"
    mistral = "mistral"
    huggingface = "huggingface"

# Pydantic model for input validation
class StoryRequest(BaseModel):
    title: str
    provider: Provider = Provider.gemini  # Default to Gemini

# Function to generate story with Gemini
def generate_with_gemini(title: str):
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini model not initialized")
    try:
        response = gemini_model.generate_content(f"Write a very short story about a {title}")
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

# Function to generate story with OpenAI
def generate_with_openai(title: str):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Write a very short story about a {title}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

# Function to generate story with Mistral
def generate_with_mistral(title: str):
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
            "messages": [{"role": "user", "content": f"Write a very short story about a {title}"}]
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

# Function to generate story with Hugging Face
def generate_with_huggingface(title: str):
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=500, detail="Hugging Face token not configured")
    
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": f"Write a very short story about a {title}",
            "parameters": {"max_new_tokens": 200}
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face error: {str(e)}")

# POST endpoint for generating a story
@app.post("/story")
async def generate_story(request: StoryRequest):
    title = request.title
    provider = request.provider
    
    try:
        if provider == Provider.gemini:
            story = generate_with_gemini(title)
        elif provider == Provider.openai:
            story = generate_with_openai(title)
        elif provider == Provider.mistral:
            story = generate_with_mistral(title)
        elif provider == Provider.huggingface:
            story = generate_with_huggingface(title)
        else:
            raise HTTPException(status_code=400, detail="Invalid provider specified")
        
        return {
            "title": title,
            "provider": provider,
            "story": story
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)