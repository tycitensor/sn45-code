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
llm_instance = submission.LLM(llm)

class CallRequest(BaseModel):
    inputs: dict

@app.post("/call")
async def call_llm(request: CallRequest):
    try:
        # Run the LLM object with the given inputs
        result = llm_instance(**request.inputs)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
