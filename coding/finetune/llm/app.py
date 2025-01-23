import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv("../../../.env")

if not os.getenv("LLM_AUTH_KEY"):
    raise ValueError("LLM_AUTH_KEY environment variable not set")



# Track token usage and current active key
token_usage: Dict[str, int] = {}
current_key: Optional[str] = None

app = FastAPI()

class InitRequest(BaseModel):
    key: str

class LLMRequest(BaseModel):
    query: str
    llm_name: str

class LLMResponse(BaseModel):
    result: str
    total_tokens: int

class EmbeddingRequest(BaseModel):
    query: str


class EmbeddingResponse(BaseModel):
    vector: List[float]

async def verify_auth(auth_key: str = Depends(lambda: os.getenv("LLM_AUTH_KEY"))):
    if not auth_key:
        raise HTTPException(
            status_code=500,
            detail="LLM_AUTH_KEY environment variable not set"
        )
    return auth_key

@app.post("/init")
async def init_key(request: InitRequest, auth_key: str = Depends(verify_auth)):
    """Initialize token tracking for a new key and set as current"""
    global current_key
    if request.key not in token_usage:
        token_usage[request.key] = 0
    current_key = request.key
    return {"message": f"Set active key to {request.key}"}

@app.post("/reset")
async def reset_count(auth_key: str = Depends(verify_auth)):
    """Reset token count for current key"""
    if not current_key:
        raise HTTPException(
            status_code=400,
            detail="No active key. Call /init endpoint first."
        )
    token_usage[current_key] = 0
    return {"message": f"Reset token count for key {current_key}"}

@app.get("/count")
async def get_count(auth_key: str = Depends(verify_auth)):
    """Get current token count"""
    if not current_key:
        raise HTTPException(
            status_code=400,
            detail="No active key. Call /init endpoint first."
        )
    return {"key": current_key, "count": token_usage[current_key]}

@app.post("/call", response_model=LLMResponse)
async def call_llm(request: LLMRequest):
    models = {
        "gpt-4o": ChatOpenAI(model="gpt-4o"),
        "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo"),
        "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini"),
        "claude-3-5-sonnet": ChatAnthropic(model="claude-3-5-sonnet-latest"),
        "gemini-2.0-flash-exp": ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp"),
    }

    global current_key
    try:
        # Verify we have an active key
        # TODO remove this when token count is implemented
        if not current_key:
        #    raise HTTPException(
        #        status_code=400,
        #        detail="No active key. Call /init endpoint first."
        #     )
            current_key = "test"
            token_usage[current_key] = 0
            
        llm = models[request.llm_name]
        
        max_retries = 50
        retry_delay = 20  # seconds
        
        for attempt in range(max_retries):
            try:
                response = llm.invoke(request.query)
                break
            except Exception as e:
                if attempt < max_retries - 1 and ("429" in str(e) or "529" in str(e)):
                    # Rate limit hit, wait and retry
                    await asyncio.sleep(retry_delay)
                    continue
                raise
        
        # Get token usage from response
        tokens = response.usage_metadata.get('total_tokens', 0)
        
        # Update token count for current key
        token_usage[current_key] += tokens
        
        return LLMResponse(
            result=response.content,
            total_tokens=token_usage[current_key]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        vector = embedder.embed_query(request.query)
        return EmbeddingResponse(vector=vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=25000)
