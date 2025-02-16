from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  
\
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Initialize Gemini LLM
# Initialize Gemini LLM
llm = ChatOpenAI(
    model="deepseek-chat",  # Change this if your model name is different
    openai_api_key=API_KEY,  # Use your actual API key
    openai_api_base="https://api.aimlapi.com/v1"  # ✅ Set custom base URL
)

# Define request model
class DocumentRequest(BaseModel):
    extracted_text: str

# Function to process text using Gemini
def analyze_text(task, text):
    print("text in function:", text)  # Debugging

    prompts = {
        "summarize": f"Summarize this document in simple terms:\n\n{text}",
        "detect_risks": f"Identify any unfair terms, privacy concerns, or legal risks in this document:\n\n{text}",
        "recommendations": f"Suggest improvements for fairness and compliance in this document:\n\n{text}"
    }

    # Create a properly formatted prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an AI-powered contract analysis assistant."),
        HumanMessage(content=prompts[task])
    ])

    formatted_messages = chat_prompt.format_messages()

    # Call Gemini LLM with correctly formatted messages
    response = llm.invoke(formatted_messages)

    # Extract the content from the response
    if hasattr(response, 'content'):
        return response.content
    else:
        raise ValueError("Invalid response format from Gemini API")

# API Route: Home page
@app.get("/")
async def home():
    return {"message": "Welcome to the Document Analysis API!"}

# API Route: Analyze document
@app.post("/analyze")
async def analyze_document(request: DocumentRequest):
    try:
        text = request.extracted_text
        print("Received extracted text:", text)  # Debugging

        # Run analysis
        summary = analyze_text("summarize", text)
        risks = analyze_text("detect_risks", text)
        recommendations = analyze_text("recommendations", text)

        return {
            "summary": summary,
            "risks": risks,
            "recommendations": recommendations
        }
    except Exception as e:
        print("Error during analysis:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI with a specified port (8000)
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", 8000))  # Use Railway's assigned port or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=PORT)
