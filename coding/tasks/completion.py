import ast
import random
from typing import Callable, List, Dict

from .task import Task
from coding.schemas import Context

def extract_random_function(code):
    """
    Takes a string of Python code, finds a random function within it, 
    and returns the function name and body as separate strings.

    Parameters:
    code (str): The Python code as a string.

    Returns:
    tuple: A tuple containing the function name and function body as separate strings.
    """
    random.seed(None)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError("Invalid Python code provided.") from e

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    if not functions:
        raise ValueError("No functions found in the provided code.")

    selected_function = random.choice(functions)

    func_name = selected_function.name
    func_body = ast.get_source_segment(code, selected_function)

    return func_name, func_body

class CompletionTask(Task):
    name: str = "completion"
    desc: str = "code completion"
    goal: str = "complete the code "
    reward_definition: List[dict] = [
        dict(name="codesim", weight=0.8),
        dict(name="speed", weight=0.2, ideal_time=0.5)
    ]
    penalty_definition: List = []
    cleaning_pipeline: List = [
    ] # TODO remove markdown wrappings
    dataset_options: Dict = {}
    attachments = []
    messages = []
    files = []
    
    def __init__(self, llm: Callable, context: Context, **kwargs):
        self.context = context

        func_name, func_body = extract_random_function(context.content) # TODO handle comments
        self.query = (
            func_name + "<|fim_hole|>" # we want them to complete that area, pretending its a hole
        )
        self.reference = func_body

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags