import re
import json
import time
import requests
from typing import Callable, List, Dict
from dataclasses import dataclass, field

from .task import Task
from coding.schemas import Context, File, Patch, Edit
from coding.rewards.reward import BatchRewardOutput, RewardEvent
from coding.rewards.codesim import CodeSimModel


def parse_diff(diff_text: str, no_title=False) -> Patch:
    diff_pattern = r"^diff --git a\/(.+?) b\/(.+?)$"
    line_change_pattern = r"^@@ -(\d+),\d+ \+(\d+),\d+ @@"
    edits = []

    current_file = None
    old_file_line_num = 0
    new_file_line_num = 0
    
    for line in diff_text.splitlines():
        diff_match = re.match(diff_pattern, line)
        if diff_match:
            current_file = diff_match.group(2)
            old_file_line_num = 0
            new_file_line_num = 0
            continue
        elif no_title and not current_file:
            current_file = ""
            old_file_line_num = 0
            new_file_line_num = 0
            continue

        line_change_match = re.match(line_change_pattern, line)
        
        if line_change_match:
            old_file_line_num = int(line_change_match.group(1))
            new_file_line_num = int(line_change_match.group(2))
            continue

        if line.startswith("+") and not line.startswith("+++"):
            # Line added in new file
            edits.append(Edit(
                file_name=current_file,
                line_number=new_file_line_num,
                line_content="",
                new_line_content=line[1:].strip()
            ))
            new_file_line_num += 1
        elif line.startswith("-") and not line.startswith("---"):
            # Line removed from old file
            edits.append(Edit(
                file_name=current_file,
                line_number=old_file_line_num,
                line_content=line[1:].strip(),
                new_line_content=""
            ))
            old_file_line_num += 1
        elif line.startswith(" "):
            # Context lines (lines present in both old and new files)
            old_file_line_num += 1
            new_file_line_num += 1

    return Patch(edits=edits)



class SWETask(Task):
    name: str = "swe"
    desc: str = "given a github issue corrrectly solve it"
    goal: str = "return the valid patch"
    reward_definition: str = [
        dict(name="speed", weight=0.1, ideal_time=25),
        dict(name="self", weight=0.9),
    ]
    penalty_definition: List = []
    cleaning_pipeline: List = []  # TODO remove markdown wrappings
    dataset_options: Dict = {}
    attachments = []
    messages = []
    files = []

    def __init__(
        self, llm: Callable, context: Context, code_scorer: Callable = None, **kwargs
    ):
        self.llm = llm
        self.context = context
        self.patch: Patch = parse_diff(context.content)
        self.query = context.topic
        self.repo = context.title
        self.base_commit = context.extras["base_commit"]
        self.pull_number = context.extras["pull_number"]

        

    def score(self, patch: Patch):
        pass
        