from swebase import LLMClient

llm = LLMClient()

passing = True
response, _ = llm("say the word happy", "gpt-4o")
if "happy" not in response:
    passing = False

response, _ = llm("say the word sad", "gpt-4o")
if "sad" not in response:
    passing = False

print("The test passed" if passing else "The test failed")