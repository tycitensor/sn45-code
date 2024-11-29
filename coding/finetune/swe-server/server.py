from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import submission

app = FastAPI()

# Initialize the LLM class from submission.py
llm_instance = submission.LLM()

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
