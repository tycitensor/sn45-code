from fastapi import FastAPI, HTTPException, Request
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os
import submission

app = FastAPI()

# Initialize the LLM class from submission.py

llm = ChatOpenAI(
        base_url="http://host.docker.internal:21000/v1",
        model_name=os.getenv("LLM_NAME"),
        temperature=0.7,
        max_tokens=16384,
    )

swe_instance = submission.SWE(llm)

class CallRequest(BaseModel):
    repo_location: str
    issue_description: str

@app.post("/call")
async def call_swe(request: CallRequest) -> dict:
    try:
        # Run the LLM object with the given inputs
        result = swe_instance(request.repo_location, request.issue_description)
        return {"result": result.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
