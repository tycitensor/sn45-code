import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List
from dotenv import load_dotenv

# ------------------------------
#    Import Provider Libraries
# ------------------------------
import openai
import anthropic
from google import genai
from langchain_openai import  OpenAIEmbeddings

# ------------------------------
#         Load Environment
# ------------------------------
load_dotenv("../../../.env")

# Set API keys (fallback to LLM_AUTH_KEY if specific keys aren’t set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_AUTH_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_AUTH_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_AUTH_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY (or LLM_AUTH_KEY) environment variable not set")

openai.api_key = OPENAI_API_KEY
anthropic_client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
google_client = genai.Client(api_key=GOOGLE_API_KEY)

# ------------------------------
#       Global Variables
# ------------------------------
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
    "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo", "max_tokens": 16384},
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 16384},
    "claude-3-5-sonnet": {"provider": "anthropic", "model": "claude-3-5-sonnet-latest", "max_tokens": 8192},
    "gemini-2.0-flash-exp": {"provider": "google", "model": "gemini-2.0-flash", "max_tokens": 8192},
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
    temperature: Optional[float] = 0.7  # default temperature value

class LLMResponse(BaseModel):
    result: str
    total_tokens: int

class EmbeddingRequest(BaseModel):
    query: str

class EmbeddingResponse(BaseModel):
    vector: List[float]

class BatchEmbeddingRequest(BaseModel):
    queries: List[str]

class BatchEmbeddingResponse(BaseModel):
    vectors: List[List[float]]
    
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
    global current_key
    if request.key not in token_usage:
        token_usage[request.key] = 0
    current_key = request.key
    return {"message": f"Set active key to {request.key}"}

@app.post("/reset")
async def reset_count(auth_key: str = Depends(verify_auth)):
    global current_key
    if not current_key:
        raise HTTPException(status_code=400, detail="No active key. Call /init endpoint first.")
    token_usage[current_key] = 0
    return {"message": f"Reset token count for key {current_key}"}

@app.get("/count")
async def get_count(auth_key: str = Depends(verify_auth)):
    global current_key
    if not current_key:
        raise HTTPException(status_code=400, detail="No active key. Call /init endpoint first.")
    return {"key": current_key, "count": token_usage[current_key]}

# ------------------------------
#   LLM Call Functions per Provider
# ------------------------------
async def call_openai(query: str, model: str, temperature: float, max_tokens: int):
    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result = response.choices[0].message.content
    tokens = response.usage.get("total_tokens", 0) if response.usage else 0
    return {"content": result, "usage": {"total_tokens": tokens}}

async def call_anthropic(query: str, model: str, temperature: float, max_tokens: int):
    # Anthropic requires a specially formatted prompt.
    prompt = f"{anthropic.HUMAN_PROMPT} {query} {anthropic.AI_PROMPT}"
    def sync_call():
        return anthropic_client.completion(
            model=model,
            prompt=prompt,
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
        )
    response = await asyncio.to_thread(sync_call)
    result = response.get("completion", "")
    tokens = response.get("usage", {}).get("total_tokens", 0)
    return {"content": result, "usage": {"total_tokens": tokens}}

async def call_google(query: str, model: str, temperature: float, max_tokens: int):
    # Using the google.genai client as per the provided example.
    def sync_call():
        # The Google API uses `contents` rather than `prompt`.
        # If the API supports temperature and max_output_tokens, pass them; otherwise, they may be ignored.
        return google_client.models.generate_content(
            model=model,
            contents=query,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    response = await asyncio.to_thread(sync_call)
    # The response is expected to have a `text` attribute (or key) containing the result.
    if hasattr(response, "text"):
        result = response.text
    elif isinstance(response, dict):
        result = response.get("text", "")
    else:
        result = ""
    return {"content": result, "usage": {"total_tokens": 0}}

async def invoke_llm(query: str, llm_config: dict, temperature: float):
    provider = llm_config["provider"]
    model = llm_config["model"]
    max_tokens = llm_config["max_tokens"]
    if provider == "openai":
        return await call_openai(query, model, temperature, max_tokens)
    elif provider == "anthropic":
        return await call_anthropic(query, model, temperature, max_tokens)
    elif provider == "google":
        return await call_google(query, model, temperature, max_tokens)
    else:
        raise ValueError("Unknown provider specified.")

# ------------------------------
#   Helper: Async LLM Invoker with Retry
# ------------------------------
async def ainvoke_with_retry(llm_config: dict, query: str, temperature: float, max_retries: int = 50, initial_delay: int = 1):
    delay = initial_delay
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = await invoke_llm(query, llm_config, temperature)
            return response
        except Exception as e:
            print("Error in ainvoke_with_retry:", e)
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
        fallback_llm = models["gpt-4o"]

        try:
            response = await ainvoke_with_retry(requested_llm, request.query, request.temperature)
        except Exception as e:
            print(f"Primary model {request.llm_name} failed, falling back to gpt-4o. Error: {e}")
            response = await ainvoke_with_retry(fallback_llm, request.query, request.temperature)

        # Update token usage (if provided by the API)
        tokens = response["usage"].get("total_tokens", 0)
        token_usage[current_key] += tokens

        return LLMResponse(
            result=response["content"],
            total_tokens=token_usage[current_key]
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
        response = await openai.Embedding.acreate(
            input=request.query,
            model="text-embedding-3-small"
        )
        vector = response["data"][0]["embedding"]
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

# ------------------------------
#      Run via Uvicorn
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=25000)
