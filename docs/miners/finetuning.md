# SWE Finetuning

## Task Outline

The task is to create a patch that fixes an issue in the repository. You will be provided the location to a repository and a description of the issue. This will be a real git repository as well as a real issue and you will be graded against the real patch. 

### What is a patch?

A patch is a list of edits to the repository. Each edit is an edit of a file, containing the file name, line number, line content, and new line content. As defined in the `Patch` class below.

```python
class Edit(BaseModel):
    file_name: str
    line_number: int
    line_content: str
    new_line_content: str

class Patch(BaseModel):
    edits: list[Edit]
```

## Things available to you

### Packages

You will have access to the modules in the `coding/constants.py` file in the `ALLOWED_MODULES` list. Along with specific imports from certain packages defined in the `coding/constants.py` file, in the `ALLOWED_IMPORTS` dictionary.

### Size Limits

You will have access to the `NUM_ALLOWED_CHARACTERS` variable in the `coding/constants.py` file. This is the maximum number of characters that can be used in your submission.

### LLM Models

You will have access to the following LLM models:

- "gpt-4o"
- "gpt-3.5-turbo"
- "gpt-4o-mini"
- "claude-3-5-sonnet"
- "gemini-2.0-flash-exp"

You will also have access to the following embedding models:

- "text-embedding-3-small"

#### How to use the models

You can use the models by calling the `llm` property of the `SWEBase` class. For example:

```python
from coding.finetune.swe-server.swebase import SWEBase

swe = SWEBase()
response, tokens = swe.llm("gpt-4o", "What is the capital of France?")
embeddings = swe.llm.embed("What is the capital of France?")
```

#### Reminders

- The server that hosts your code is restricted to not allow for internet access. You should not try to use it as you will likely fail.

## Submission

Locate the `coding/miners/swe.py` file. This is where your miner will go to grab your submission.

Your submission must initiate a class `SWE` that inherits from `SWEBase`. This will be called with a `repo_location` and `issue_description`. 

The `SWE` class must return a `Patch` object. This will be used to evaluate your submission.

## Testing

Use the notebook `notebooks/sample-swe-task.ipynb` to test your submission.

You need to verify your logic using the notebook `notebooks/logic-verification.ipynb`. 
