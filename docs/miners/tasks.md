# Coding Tasks

### 1. Completion

The goal of this task is to complete the given code. You will be provided a functions name and must complete the function body.

The only protocol being sent is `query`, the expected response is the completed function body.


### 2. Debugging

This task is under development.

### 3. Fill-In-The-Middle (FIM)

The goal of this task is to fill in the middle of the given code. You will be provided a portion of code with a chunk missing. The chunk to be filled in is marked with "<fim_hole>". You should return the code to be placed in the filled in spot.

The only protocol being sent is `query`, the expected response is the code to be placed in the "<fim_hole>".


### 4. Organic Convo

This task is dynamic and will be at random sent using the input from the user using the frontend. You will be sent a conversation from the chat-frontend and are expected to return a good response.

You will be provided `messages` and potentially some `files`. You must return with an appropriate response given the messages and files.


### 5. Repo

In this task you will be sent a `query` containing a majority of the code from a file in a given repo, alongside that you will be given `files` containing the other files in the repo. Your goal is to use the files to complete the missing code in the query file.


### 6. Repo File 

In this task you will be given a `query` containing a summary of what a python file did, and `files` containing some other files that came from the same repo. You are to write the entire python file given the summary and files. 
