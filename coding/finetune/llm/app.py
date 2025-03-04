from dotenv import load_dotenv

load_dotenv("../../../.env", override=False)  # Don't override existing env vars

import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List

# ------------------------------
#    Import Provider Libraries
# ------------------------------
import openai
import anthropic
from google import genai
from google.genai import types
from langchain_openai import OpenAIEmbeddings


openai.base_url = "https://openrouter.ai/api/v1"

token_usage: Dict[str, int] = {}
current_key: Optional[str] = None

# FastAPI App
app = FastAPI()

# Instead of LangChain instances we now map model names to a config that includes:
# - provider: one of "openai", "anthropic", "google"
# - model: the actual model name/ID used by the API
# - max_tokens: maximum tokens to request (used for each API call)
models = {
    "gpt-4o": {"provider": "openai", "model": "gpt-4o", "max_tokens": 16384},
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "max_tokens": 16384,
    },
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 16384},
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 8192,
    },
    "gemini-2.0-flash-exp": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "max_tokens": 8192,
    },
}
embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))


class InitRequest(BaseModel):
    key: str


class LLMRequest(BaseModel):
    query: str
    api_key: str
    llm_name: str
    temperature: Optional[float] = 0.7  # default temperature value


class LLMResponse(BaseModel):
    result: str
    total_tokens: int


class EmbeddingRequest(BaseModel):
    query: str


class EmbeddingResponse(BaseModel):
    vector: List[float]


# New models for batch embedding support
class BatchEmbeddingRequest(BaseModel):
    queries: List[str]


class BatchEmbeddingResponse(BaseModel):
    vectors: List[List[float]]


class BatchEmbeddingResponse(BaseModel):
    vectors: List[List[float]]


# ------------------------------
#       Auth Dependency
# ------------------------------
async def verify_auth(auth_key: str = Depends(lambda: os.getenv("LLM_AUTH_KEY"))):
    if not auth_key:
        raise HTTPException(
            status_code=500, detail="LLM_AUTH_KEY environment variable not set"
        )
    return auth_key


# ------------------------------
#   Initialize / Reset / Count
# ------------------------------
@app.post("/init")
async def init_key(request: InitRequest, auth_key: str = Depends(verify_auth)):
    global current_key
    if request.key not in token_usage:
        token_usage[request.key] = 0
    current_key = request.key
    return {"message": f"Set active key to {request.key}"}


@app.post("/reset")
async def reset_count(auth_key: str = Depends(verify_auth)):
    global current_key
    if not current_key:
        raise HTTPException(
            status_code=400, detail="No active key. Call /init endpoint first."
        )
    token_usage[current_key] = 0
    return {"message": f"Reset token count for key {current_key}"}


@app.get("/count")
async def get_count(auth_key: str = Depends(verify_auth)):
    global current_key
    if not current_key:
        raise HTTPException(
            status_code=400, detail="No active key. Call /init endpoint first."
        )
    return {"key": current_key, "count": token_usage[current_key]}


async def call_openai(
    query: str, model: str, temperature: float, max_tokens: int = 16384, api_key: str = None, provider: str = "openai"
):
    openai.api_key = api_key
    def sync_call():
        response = openai.chat.completions.create(
            model=f"{provider}/{model}",
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response

    response = await asyncio.to_thread(sync_call)
    result = response.choices[0].message.content
    tokens = response.usage.completion_tokens + response.usage.prompt_tokens
    return {"content": result, "usage": {"total_tokens": tokens}}

async def ainvoke_with_retry(
    llm_config: dict,
    query: str,
    temperature: float,
    api_key: str,
    max_retries: int = 50,
    initial_delay: int = 1,
):
    delay = initial_delay
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = await call_openai(query, llm_config["model"], temperature, llm_config["max_tokens"], api_key, llm_config["provider"])
            return response
        except Exception as e:
            print(
                "Error in ainvoke_with_retry:", e, "when calling", llm_config["model"]
            )
            # Retry on rate-limit or server errors
            if "429" in str(e) or "529" in str(e):
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise
            else:
                raise
    if last_exception:
        raise last_exception
    else:
        raise HTTPException(status_code=500, detail="Unknown error invoking LLM")


# ------------------------------
#          Call LLM Endpoint
# ------------------------------
@app.post("/call", response_model=LLMResponse)
async def call_llm(request: LLMRequest):
    global current_key, token_usage
    try:
        if not current_key:
            # If no key is initialized, default to "test"
            current_key = "test"
            token_usage[current_key] = 0

        # Attempt the requested LLM; fall back to "gpt-4o" if not found.
        requested_llm = models.get(request.llm_name, models["gpt-4o"])

        response = await ainvoke_with_retry(
            requested_llm, request.query, request.temperature, request.api_key
        )

        # Update token usage (if provided by the API)
        tokens = response["usage"].get("total_tokens", 0)
        token_usage[current_key] += tokens

        return LLMResponse(
            result=response["content"], total_tokens=token_usage[current_key]
        )
    except Exception as e:
        print("Error in call_llm endpoint:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
#          Embedding Endpoint
# ------------------------------
@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """
    Get embeddings for the given query using OpenAI's embeddings API.
    """
    try:
        response = await embedder.aembed_query(request.query)
        vector = response
        return EmbeddingResponse(vector=vector)
    except Exception as e:
        print("Error in get_embeddings endpoint:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
#      Batch Embeddings
# ------------------------------
@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def get_batch_embeddings(request: BatchEmbeddingRequest):
    """
    Returns embedding vectors for a batch of input queries.
    """
    try:
        # Run embedding tasks concurrently for all queries in the batch.
        vectors = await embedder.aembed_documents(request.queries)
        return BatchEmbeddingResponse(vectors=vectors)
    except Exception as e:
        print("An error occurred in get_batch_embeddings", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
#      Run via Uvicorn
# ------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=25000)
