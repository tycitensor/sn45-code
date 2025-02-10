import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List
from dotenv import load_dotenv

# ------------------------------
#  LangChain-based LLM Imports
# ------------------------------
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv("../../../.env")

if not os.getenv("LLM_AUTH_KEY"):
    raise ValueError("LLM_AUTH_KEY environment variable not set")


# ------------------------------
#      Global Variables
# ------------------------------
token_usage: Dict[str, int] = {}
current_key: Optional[str] = None

# FastAPI App
app = FastAPI()

models = {
        "gpt-4o": ChatOpenAI(model="gpt-4o", max_tokens=16384),
        "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo", max_tokens=16384),
        "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", max_tokens=16384),
        "claude-3-5-sonnet": ChatAnthropic(model="claude-3-5-sonnet-latest", max_tokens=8192),
        "gemini-2.0-flash-exp": ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", max_tokens=8192),
    }
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ------------------------------
#       Pydantic Models
# ------------------------------
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


# ------------------------------
#       Auth Dependency
# ------------------------------
async def verify_auth(auth_key: str = Depends(lambda: os.getenv("LLM_AUTH_KEY"))):
    if not auth_key:
        raise HTTPException(
            status_code=500,
            detail="LLM_AUTH_KEY environment variable not set"
        )
    return auth_key


# ------------------------------
#   Initialize / Reset / Count
# ------------------------------
@app.post("/init")
async def init_key(request: InitRequest, auth_key: str = Depends(verify_auth)):
    """Initialize token tracking for a new key and set as current."""
    global current_key
    if request.key not in token_usage:
        token_usage[request.key] = 0
    current_key = request.key
    return {"message": f"Set active key to {request.key}"}

@app.post("/reset")
async def reset_count(auth_key: str = Depends(verify_auth)):
    """Reset token count for current key."""
    global current_key
    if not current_key:
        raise HTTPException(
            status_code=400,
            detail="No active key. Call /init endpoint first."
        )
    token_usage[current_key] = 0
    return {"message": f"Reset token count for key {current_key}"}

@app.get("/count")
async def get_count(auth_key: str = Depends(verify_auth)):
    """Get current token count."""
    global current_key
    if not current_key:
        raise HTTPException(
            status_code=400,
            detail="No active key. Call /init endpoint first."
        )
    return {"key": current_key, "count": token_usage[current_key]}


# ------------------------------
#   Helper: Async LLM Invoker
# ------------------------------
async def ainvoke_with_retry(llm, query: str, max_retries: int = 50, initial_delay: int = 1):
    """
    Invoke the LLM asynchronously with exponential backoff on rate-limit or server errors.
    Returns the response if successful; raises Exception after max_retries.
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke(query)
            return response
        except Exception as e:
            print("An error occurred in ainvoke_with_retry", e)
            # Check if it's a rate-limit or server error
            if "429" in str(e) or "529" in str(e):
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    # Retries exhausted
                    raise
            else:
                # Some other error - don't keep retrying
                raise

    # If we exit the loop without returning, raise whatever last exception we had
    if last_exception:
        raise last_exception
    else:
        raise HTTPException(status_code=500, detail="Unknown error invoking LLM")


# ------------------------------
#          Call LLM
# ------------------------------
@app.post("/call", response_model=LLMResponse)
async def call_llm(request: LLMRequest):
    """Call one of the registered LLMs. If repeated failures, fallback to 'gpt-4o'."""
    global current_key, token_usage


    try:
        if not current_key:
            # If no key was initialized, default to "test" so code doesn't break
            current_key = "test"
            token_usage[current_key] = 0

        # Try to retrieve requested model; fallback to "gpt-4o" if not found
        requested_llm = models.get(request.llm_name, models["gpt-4o"])
        fallback_llm = models["gpt-4o"]

        # --- Step 1: Try the requested LLM ---
        try:
            response = await ainvoke_with_retry(requested_llm, request.query)
        except Exception:
            # If the requested LLM fails after max retries, fallback
            response = await ainvoke_with_retry(fallback_llm, request.query)

        # Extract tokens from usage metadata (some LLMs may not provide it)
        tokens = response.usage_metadata.get("total_tokens", 0)

        # Update token usage
        token_usage[current_key] += tokens

        return LLMResponse(
            result=response.content,
            total_tokens=token_usage[current_key]
        )
    except Exception as e:
        print("An error occurred in call_llm", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
#        Embeddings
# ------------------------------
@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """
    Returns embeddings vector for the given input query.
    """
    try:
        # embed_query is often synchronous in many libraries; if there's an async version, use that instead.
        vector = await embedder.aembed_query(request.query)
        return EmbeddingResponse(vector=vector)
    except Exception as e:
        print("An error occurred in get_embeddings", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
#      Run via Uvicorn
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=25000)
