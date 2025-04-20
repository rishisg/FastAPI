# FastAPI Project - Text Generation, Summarization, and Translation

Project Overview
A FastAPI-powered web service that generates short stories based on user-provided titles. Supports multiple AI models, including Gemini and Mistral, for diverse storytelling styles.

⚙️ Technologies Used
FastAPI: For building the web API.

Pydantic: For data validation.

Uvicorn: As the ASGI server.

Gemini / Mistral: AI models for sStory Generation, Summarize and Translation.

Python-dotenv: For environment variable management.

Requests: For making HTTP requests.

pip (Python package installer)​

🛠️ Installation & Setup
Prerequisites
Python 3.8+

The three distinct FastAPI applications that provide different functionalities:
1. **Text Generation**: Generate text using AI models.
2. **Summarization**: Summarize long pieces of text.
3. **Translation**: Translate text between languages.

Each service is configured to interact with multiple AI providers, including **Gemini** and **Mistral**. You can easily switch between these providers by specifying the provider in the API requests.

## Features

- **Text Generation**: Uses the OpenAI API or Gemini model to generate text based on prompts.
- **Summarization**: Uses Gemini or Mistral to summarize large text.
- **Translation**: Uses Gemini or Mistral to translate text between different languages.


### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/your-username/fastapi-project.git

2. Setup Python Virtual Environment
Navigate to the appropriate directory for each service (Textgeneration, Summarisation, Translation) and set up the virtual environment.

For example, for Textgeneration:
2. Setup Python Virtual Environment
Navigate to the appropriate directory for each service (Textgeneration, Summarisation, Translation) and set up the virtual environment.

For example, for Textgeneration:
2. Setup Python Virtual Environment
Navigate to the appropriate directory for each service (Textgeneration, Summarisation, Translation) and set up the virtual environment.

For example, for Textgeneration:
cd FastAPI/Textgeneration
python -m venv venv

Activate the virtual environment:
Windows:
venv\Scripts\activate

3. Install Dependencies
Install the required dependencies:
pip install -r requirements.txt

4. Create .env File
Create a .env file in the root of the directory and add the following variables with your credentials:
API_USERNAME=your-username
API_PASSWORD=your-password
MISTRAL_API_KEY=your-mistral-api-key
GEMINI_API_KEY=your-gemini-api-key

5. Run the Application
Run the FastAPI application for each service using uvicorn:
uvicorn main:app --reload

The service will be available at http://127.0.0.1:8000.
Swagger documentation will be accessible at http://127.0.0.1:8000/docs.

Authentication
Each service requires Basic Authentication. You must pass the correct username and password when accessing the endpoints. These credentials are read from the .env file.

Example Request:
For Text Generation, you can make a POST request to /story with a JSON body:

#Text Generation: /story
{
 "title": "cat",
 "provider": "gemini"
}

#Summarization:/summarize
{
  "text": "This is a long paragraph of text that needs to be summarized. It could be a lengthy article, blog post, or anything want to short.",
  "provider": "gemini"
}

#Translation: /translate
{
  "text": "Hello, how are you?",
  "source_language": "en",
  "target_language": "fr",
  "provider": "gemini"
}


Testing APIs in Postman:
-----------------------
1. Create a new request:
Open Postman.
Click on "New" and select "Request".
Give your request a name, and save it to a new collection (you can name the collection according to your project, e.g., "FastAPI Project").

2. Testing Authentication
Since your FastAPI apps require Basic Authentication, you need to add the username and password to the request.
In the Authorization tab of Postman:
Select Basic Auth from the drop-down.
Enter the API_USERNAME and API_PASSWORD from the .env file.

3. Testing APIs
To test the Textgeneraion, Summarisation and Translation functionalities, create a POST request.
URL: Assuming your FastAPI app is running locally on port 8000 (default):
http://127.0.0.1:8000/story or summarize or translate 

Body: In the Body tab, select raw and set the format to JSON. Then add the following JSON payload to test the API's:

#:Text Generation
{
 "title": "cat",
 "provider": "gemini"
}

#Summarization:
{
  "text": "This is a long paragraph of text that needs to be summarized. It could be a lengthy article, blog post, or anything want to short.",
  "provider": "gemini"
}

#Translation
{
  "text": "Hello, how are you?",
  "source_language": "en",
  "target_language": "fr",
  "provider": "gemini"
}

4. Sending the request: Click Send, and Postman will show the output in the response.


Testing APIs Using cURL:
-----------------------

1. Testing Text Generation API
If you are testing a text generation endpoint, the cURL command will look like this:
Command:
curl -u your_username:your_password -X POST http://127.0.0.1:8000/textgeneration \
-H "Content-Type: application/json" \
-d '{"prompt": "Write a short story about a robot.", "provider": "openai"}'

2. Testing Summarization API
For the Summarization API, you can use a similar cURL request:
Command:
curl -u your_username:your_password -X POST http://127.0.0.1:8000/summarize \
-H "Content-Type: application/json" \
-d '{"text": "This is a long paragraph that needs to be summarized.", "provider": "gemini"}'

3.Testing Translation API
To test the Translation API using cURL, you can send a POST request with JSON data:
Command:
curl -u your_username:your_password -X POST http://127.0.0.1:8000/translate \
-H "Content-Type: application/json" \
-d '{"text": "Hello, how are you?", "source_language": "en", "target_language": "fr", "provider": "gemini"}'