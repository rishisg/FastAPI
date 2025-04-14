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

# Configure OpenAI
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

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str
    provider: Provider = Provider.gemini  # Default provider

def translate_with_gemini(text: str, source: str, target: str) -> str:
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini model not initialized")
    try:
        prompt = f"""Translate the following text from {source} to {target}.
        Preserve all formatting and special characters.
        Return only the translated text without additional commentary.
        
        Text to translate: "{text}\""""
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

def translate_with_openai(text: str, source: str, target: str) -> str:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a professional translator. Translate from {source} to {target} while preserving meaning, tone, and formatting."
                },
                {
                    "role": "user", 
                    "content": f"Translate this text exactly without additions:\n\n{text}"
                }
            ],
            temperature=0.2,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

def translate_with_mistral(text: str, source: str, target: str) -> str:
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise HTTPException(status_code=500, detail="Mistral API key not configured")
    
    try:
        headers = {
            "Authorization": f"Bearer {mistral_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistral-medium",
            "messages": [
                {
                    "role": "user", 
                    "content": f"Translate this exactly from {source} to {target} without commentary:\n\n{text}"
                }
            ],
            "temperature": 0.3
        }
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Mistral API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mistral error: {str(e)}")

def translate_with_huggingface(text: str, source: str, target: str) -> str:
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=500, detail="Hugging Face token not configured")
    
    try:
        API_URL = f"https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-{source}-{target}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": text}
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        # Handle Hugging Face's async model loading
        if response.status_code == 503:
            estimated_time = response.json().get("estimated_time", 30)
            raise HTTPException(
                status_code=503,
                detail=f"Model is loading, please try again in {estimated_time} seconds"
            )
        
        response.raise_for_status()
        return response.json()[0]["translation_text"]
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Hugging Face API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face error: {str(e)}")

@app.post("/translate/")
async def translate(request: TranslationRequest):
    provider = request.provider
    text = request.text
    source = request.source_language
    target = request.target_language
    
    try:
        if provider == Provider.gemini:
            translated_text = translate_with_gemini(text, source, target)
        elif provider == Provider.openai:
            translated_text = translate_with_openai(text, source, target)
        elif provider == Provider.mistral:
            translated_text = translate_with_mistral(text, source, target)
        elif provider == Provider.huggingface:
            translated_text = translate_with_huggingface(text, source, target)
        else:
            raise HTTPException(status_code=400, detail="Invalid provider specified")
        
        return {
            "provider": provider.value,
            "source_language": source,
            "target_language": target,
            "original_text": text,
            "original_length": len(text),
            "translated_text": translated_text,
            "translated_length": len(translated_text)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)