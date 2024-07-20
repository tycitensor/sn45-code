import ast
import time
import json
import traceback
from typing import Callable, List, Dict, Tuple

from .task import Task
from coding.schemas import Context, File, ObscurePackage
from coding.repl import REPLClient, PackageInfo
from coding.helpers import extract_python_code
from coding.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
    RewardEvent
)

def find_used_objects(script: str, package_name: str) -> List[str]:
    class ImportVisitor(ast.NodeVisitor):
        def __init__(self):
            self.used_objects = []
            self.package_name = package_name

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == self.package_name:
                    self.used_objects.append(alias.name)
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module == self.package_name:
                for alias in node.names:
                    self.used_objects.append(alias.name)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == self.package_name:
                self.used_objects.append(node.attr)
            self.generic_visit(node)

    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        print(f"Syntax error while parsing the script: {e}")
        return []

    visitor = ImportVisitor()
    visitor.visit(tree)
    return visitor.used_objects

def break_declarations(script: str, files: List[File], used_objects: List[str]) -> Tuple[List[File], List[File]]:
    class FunctionDefVisitor(ast.NodeVisitor):
        def __init__(self, objects_to_break):
            self.objects_to_break = objects_to_break
            self.lines_to_break = []

        def visit_FunctionDef(self, node):
            if node.name in self.objects_to_break:
                self.lines_to_break.append(node.lineno)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            if node.name in self.objects_to_break:
                self.lines_to_break.append(node.lineno)
            self.generic_visit(node)

        def visit_Assign(self, node):
            if any(isinstance(target, ast.Name) and target.id in self.objects_to_break for target in node.targets):
                self.lines_to_break.append(node.lineno)
            self.generic_visit(node)

    def break_line(content: str, lineno: int) -> str:
        lines = content.split('\n')
        lines[lineno - 1] = '# BROKEN: ' + lines[lineno - 1]
        return '\n'.join(lines)

    updated_files = []
    broken_files = []

    for file in files:
        try:
            file_tree = ast.parse(file.content)
        except SyntaxError as e:
            print(f"Syntax error while parsing the file {file.path}: {e}")
            updated_files.append(file)
            continue

        file_visitor = FunctionDefVisitor(used_objects)
        file_visitor.visit(file_tree)

        if file_visitor.lines_to_break:
            broken_files.append(file)
            for lineno in file_visitor.lines_to_break:
                file.content = break_line(file.content, lineno)

        updated_files.append(file)

    return updated_files, broken_files

def gen_example_use(llm: Callable, package_name: str):
    for _ in range(20):
        example = extract_python_code(
            # llm.invoke(f"Provide an example script that uses the python package {package_name}. Do not use any extra python packages that do not come preinstalled. The script should not depend on any extra dependencies, it should be able to be ran in a brand new environment with only the package installed. Your response should be python wrapped in markdown. The script must actually use the package, do not just import it.").content
            llm.invoke(f"""Write an example Python script using the `{package_name}` package. Ensure the script is wrapped in markdown backticks and does not use any other packages besides those inbuilt to Python.
                       
The script should be an example use of the `{package_name}` package. Demonstrating the common use of the package. The package should print a result at the end.

```python
# Example script using the {package_name} package
```""").content
        )
        print(f"🍌🍌🍌🍌🍌🍌 {example}")
        if example:
            return example[0]



# TODO:
# remove comments
# Clean this up a LOT
class DebugTask(Task):
    name: str = "debug"
    desc: str = "fix the error in the code"
    goal: str = "to fix the error in a given code"
    reward_definition: str = [
        dict(name="codesim", weight=0.45),
        dict(name="speed", weight=0.1, ideal_time=5),
        dict(name="self", weight=0.45),
    ]
    penalty_definition: List = []
    cleaning_pipeline: List = []  # TODO remove markdown wrappings
    dataset_options: Dict = {}
    attachments = []
    messages = []
    files = []
    
    
    def __init__(self, llm: Callable, context: Context, repl: REPLClient):
        self.llm = llm
        self.repl = repl
        self.context = context
        self.context.files = repl.get_package_code(package_info=PackageInfo(name=context.title))
        print(f"==CHOSEN PIP PACKAGE == {context.title}")
        script = gen_example_use(llm, context.title)
        
        repl_output = repl.run_and_delete(
            context.title, updated_files=[], script=script
        )
        # if repl_output['message'] == "Fail": # TODO add this back? sometimes theres errors that are acceptable for comparison
            # raise Exception("Failed to run the script")
        self.correct_output = repl_output['output'] 

        self.package = ObscurePackage(files=self.context.files)
        self.package.obscure_package()
        script = self.package.obscure_string(script)
        
        self.query = script
        
        
        files = self.package.files 
        
        used_objects = find_used_objects(script, context.title)
        updated_files, broken_files = break_declarations(script, files, used_objects)
        if len(broken_files) == 0:
            raise Exception("Failed to generate some broken files.")
        self.files = broken_files
        self.reference = dict(
            codesim=updated_files[0].content,
        )
        
        
        
        self.attachments = [{"error": repl.run_and_delete(
            context.title, updated_files=broken_files, script=script
        )['output'], "files": broken_files}]
        

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
    
    def score(self, completion):
        print(f"getting debug score  ❎ ❎ ❎ ❎ ❎ ❎")
        print("Completion", completion)
        if not completion:
            return 0
        try:
            json.loads(completion)
        except:
            return 0
        self.package.update_file(File(**json.loads(completion))) # TODO completion should be a list of files
        self.package.undo_obscure_package()
        miner_output = self.repl.run_and_delete(
            self.context.title, updated_files=self.package.files, script=self.package.undo_obscure_string(self.query)
        )['output']
        print(f"💰💰💰💰\n Correct output: {self.correct_output}\nMiner output: {miner_output}") 
        if self.correct_output != miner_output:
            return 0
        
    
    def reward(self, completions: List[dict]) -> BatchRewardOutput:
        """Get the score between two strings.
        """

        rewards = []
        timings = []

        for completion in completions:
            t0 = time.time()
            rewards.append(self.score(completion))
            timings.append(time.time() - t0)
        output = BatchRewardOutput(
            rewards=rewards,
            timings=timings,
            extra_info={}
        )

        return output
    
    def reward_apply(self, response_event, reward_type) -> RewardEvent:
        t0 = time.time()
        batch_rewards_output = self.reward(response_event.completions)
        batch_rewards_time = time.time() - t0
        
        return RewardEvent(
            model_name=self.name,
            rewards=batch_rewards_output.rewards,
            rewards_normalized=batch_rewards_output.rewards_normalized,
            model_type=reward_type,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
            timings=batch_rewards_output.timings,
        )
    