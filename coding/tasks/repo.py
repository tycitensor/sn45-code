import ast
import random
from typing import Callable, List, Dict

from .task import Task
from coding.schemas import Context, File

# TODO only works with python, also make it so that theres a limit on the stuff it deletes, it could delete a 1000 line file.
def delete_function_body_and_following(code: str) -> (str, str):
    """
    Takes in some code, randomly finds a function, deletes the body of that function and anything after it
    
    Returns the modified code alognside the deleted body of the function 
    """
    random.seed(None)
    class FunctionBodyRemover(ast.NodeTransformer):
        def __init__(self):
            self.target_func_name = None
            self.body = None

        def visit_FunctionDef(self, node):
            if self.target_func_name is None:
                self.target_func_name = node.name
                self.body = ast.unparse(node.body)
                node.body = []
            return node

    # Parse the code into an AST
    tree = ast.parse(code)

    # Randomly select a function to delete the body from
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return code, ""

    target_func = random.choice(functions)

    # Remove the body of the target function
    remover = FunctionBodyRemover()
    remover.target_func_name = target_func.name
    remover.visit(tree)

    # Get the modified code
    modified_code = ast.unparse(tree)

    # Get the code before the target function definition
    func_def_start = code.find(f'def {target_func.name}')
    code_before_func = code[:func_def_start]

    # Combine the modified code and the part before the target function
    final_code = code_before_func + modified_code

    return final_code, remover.body

class RepoCompletionTask(Task):
    name: str = "repo"
    desc: str = "repository level code completion"
    goal: str = "complete the code given the context of the rest of the repo"
    reward_definition: List[dict] = [
        dict(name="codesim", weight=0.8),
        dict(name="speed", weight=0.2, ideal_time=1)
    ]
    penalty_definition: List = []
    cleaning_pipeline: List = [
    ] # TODO remove markdown wrappings
    dataset_options: Dict = dict(include_sibling_docs=True)
    attachments = []
    messages = []
    files = []    

    def __init__(self, llm: Callable, context: Context, **kwargs):
        self.context = context

        mod_code, correct_body = delete_function_body_and_following(context.content)
        self.query = (
            mod_code + "<|fim_hole|>"
        )
        self.files = [File(path=cont.title, content=cont.content) for cont in context.extras['sibling_docs']] # Filter the info sent to the miners
        self.reference = correct_body

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags