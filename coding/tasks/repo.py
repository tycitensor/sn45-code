import ast
import random
from typing import Callable, List, Dict

from .task import Task
from coding.schemas import Context, File
from coding.helpers.fim import insert_fim_hole

def delete_function_body_and_following(code: str) -> (str, str):
    """
    Takes in some code, randomly finds a function, deletes the body of that function and anything after it.
    
    Returns the function definition alongside the deleted body of the function.
    """
    random.seed(None)
    
    class FunctionBodyRemover(ast.NodeTransformer):
        def __init__(self, target_func_name):
            self.target_func_name = target_func_name
            self.body = None
            self.stop_processing = False

        def visit_FunctionDef(self, node):
            if self.stop_processing:
                return None
            if node.name == self.target_func_name:
                self.body = ast.unparse(node.body) if node.body else ""
                node.body = []  # Remove the function body
                self.stop_processing = True  # Stop after we modify the targeted function
            return node

    # Parse the code into an ASTt
    try:
        tree = ast.parse(code)
    except Exception as e:
        return None, None

    # Randomly select a function to delete the body from
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None, None

    target_func = random.choice(functions)

    # Remove the body of the target function
    remover = FunctionBodyRemover(target_func.name)
    remover.visit(tree)

    # If the body was not captured, return an empty string
    if remover.body is None or remover.body.strip() == "":
        return None, None

    # Find the function definition line in the original code
    func_def_start = code.find(f'def {target_func.name}')
    
    if func_def_start == -1:
        return None, None

    # Extract just the function definition line
    func_def_end = code.find(":", func_def_start) + 1
    function_definition = code[func_def_start:func_def_end]
    
    if function_definition.strip() == "":
        return None, None
    
    if not function_definition or not remover.body:
        return None, None
        
    return function_definition, remover.body


class RepoCompletionTask(Task):
    name: str = "repo"
    desc: str = "repository level code completion"
    goal: str = "complete the code given the context of the rest of the repo"
    reward_definition: List[dict] = [
        dict(name="codesim", weight=0.8),
        dict(name="speed", weight=0.2, ideal_time=2.5)
    ]
    penalty_definition: List = [
        dict(name="validcode", weight=1) 
    ]
    cleaning_pipeline: List = [
    ] # TODO remove markdown wrappings
    dataset_options: Dict = dict(include_sibling_docs=True)
    attachments = []
    messages = []
    files = []    

    def __init__(self, llm: Callable, context: Context, **kwargs):
        self.context = context

        mod_code, correct_body = delete_function_body_and_following(context.content)
        if mod_code is None or correct_body is None:
            self.query, self.reference = insert_fim_hole(context.content)
        else:
            self.query = (
                mod_code + "<|fim_hole|>"
            )
            self.reference = correct_body
        self.files = [File(path=cont.title, content=cont.content) for cont in context.extras['sibling_docs']] # Filter the info sent to the miners

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags