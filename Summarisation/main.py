from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import google.generativeai as genai
from openai import OpenAI
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize model clients
gemini_model = None
openai_client = None

# Configure providers
try:
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Gemini initialization error: {e}")

try:
    if os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"OpenAI initialization error: {e}")

class Provider(str, Enum):
    gemini = "gemini"
    openai = "openai"
    mistral = "mistral"
    huggingface = "huggingface"

class SummarizeRequest(BaseModel):
    text: str
    provider: Provider = Provider.gemini  # Default provider

def summarize_with_gemini(text: str):
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini model not initialized")
    try:
        response = gemini_model.generate_content(
            f"Please provide a concise summary of the following text:\n\n{text}"
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

def summarize_with_openai(text: str):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize this text in 3-5 bullet points:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

def summarize_with_mistral(text: str):
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise HTTPException(status_code=500, detail="Mistral API key not configured")
    
    try:
        headers = {
            "Authorization": f"Bearer {mistral_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistral-tiny",
            "messages": [
                {"role": "user", "content": f"Summarize this text concisely in one paragraph:\n\n{text}"}
            ],
            "temperature": 0.4
        }
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mistral error: {str(e)}")

def summarize_with_huggingface(text: str):
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=500, detail="Hugging Face token not configured")
    
    try:
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": text,
            "parameters": {"max_length": 130, "min_length": 30}
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]["summary_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face error: {str(e)}")

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    provider = request.provider
    text = request.text
    
    try:
        if provider == Provider.gemini:
            summary = summarize_with_gemini(text)
        elif provider == Provider.openai:
            summary = summarize_with_openai(text)
        elif provider == Provider.mistral:
            summary = summarize_with_mistral(text)
        elif provider == Provider.huggingface:
            summary = summarize_with_huggingface(text)
        else:
            raise HTTPException(status_code=400, detail="Invalid provider specified")
        
        return {
            "provider": provider,
            "original_text_length": len(text),
            "summary_length": len(summary),
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)