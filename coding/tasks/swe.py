import re
import json
import time
import requests
from typing import Callable, List, Dict
from dataclasses import dataclass, field

from .task import Task
from coding.schemas import Context, File
from coding.rewards.reward import BatchRewardOutput, RewardEvent
from coding.rewards.codesim import CodeSimModel


# TODO decide if should move diff stuff and downloading to dataset
@dataclass
class Diff:
    file: str
    edited_lines: List[tuple] = field(
        default_factory=list
    )  # +/- , line number , line content


def parse_diff(diff_text: str, no_title=False) -> List[Diff]:
    diff_pattern = r"^diff --git a\/(.+?) b\/(.+?)$"
    line_change_pattern = r"^@@ -\d+,\d+ \+(\d+),(\d+) @@"
    diff_objects = []

    current_diff = None
    current_line_num = 0

    for line in diff_text.splitlines():
        diff_match = re.match(diff_pattern, line)
        if diff_match:
            if current_diff:
                diff_objects.append(current_diff)
            current_diff = Diff(file=diff_match.group(2))
            current_line_num = 0
            continue
        elif no_title:
            current_diff = Diff(file="")
            current_line_num = 0
            continue

        line_change_match = re.match(line_change_pattern, line)
        if line_change_match:
            current_line_num = int(line_change_match.group(1))
            continue

        if line.startswith("+") and not line.startswith("+++"):
            current_diff.edited_lines.append(("+", current_line_num, line[1:]))
            current_line_num += 1
        elif line.startswith("-") and not line.startswith("---"):
            current_diff.edited_lines.append(("-", current_line_num, line[1:]))
            # The line number does not increase for deleted lines
        elif not line.startswith((" ", "-", "+", "@@", "diff")):
            current_line_num += 1

    if current_diff:
        diff_objects.append(current_diff)

    return diff_objects


import requests


def download_git_file(repo_name, pull_number, file_path):
    # Construct the base URL for the GitHub repository's pulls
    pulls_url = f"https://api.github.com/repos/{repo_name}/pulls/{pull_number}"

    # Get the pull request info to extract the commit SHA
    pull_response = requests.get(pulls_url)

    if pull_response.status_code == 200:
        pull_info = pull_response.json()
        commit_sha = pull_info["head"]["sha"]

        # Construct the URL for the file in the specific commit
        file_url = f"https://api.github.com/repos/{repo_name}/contents/{file_path}?ref={commit_sha}"
        file_response = requests.get(file_url)

        if file_response.status_code == 200:
            # The response is JSON containing file info, including the download URL
            file_info = file_response.json()
            download_url = file_info["download_url"]

            # Fetch the file content
            content_response = requests.get(download_url)

            if content_response.status_code == 200:
                # Return the file content
                return content_response.content
            else:
                raise Exception(
                    f"Failed to download {file_path} from commit {commit_sha}. HTTP Status Code: {content_response.status_code}"
                )
        else:
            raise Exception(
                f"Failed to get file info for {file_path} from commit {commit_sha}. HTTP Status Code: {file_response.status_code}"
            )
    else:
        raise Exception(
            f"Failed to get pull request info for #{pull_number} from {repo_name}. HTTP Status Code: {pull_response.status_code}"
        )


class SWETask(Task):
    name: str = "swe"
    desc: str = "given a github issue corrrectly solve it"
    goal: str = "return the valid patch"
    reward_definition: str = [
        dict(name="speed", weight=0.1, ideal_time=10),
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
        self.diffs: List[Diff] = parse_diff(context.content)

        self.context.files = [
            File(
                path=diff.file,
                content=download_git_file(
                    context.title, context.extras["pull_number"], diff.file
                ),
            )
            for diff in self.diffs
        ]

        # renaming the files
        # for idx, file in enumerate(self.diffs):
        #     id = str(uuid.uuid4())[0:5]
        #     self.diffs[idx].file = id
        #     self.context.files[idx].path = id

        self.files = self.context.files
        self.query = (
            """Given the following issue and files, please return a patch file that would fix the issue. An example of what you should return is
<patch> diff --git a/example.txt b/example.txt
index e69de29..d95f3ad 100644
--- a/example.txt
+++ b/example.txt
@@ -1,3 +1,3 @@
-Hello, world!
+Hello, universe!
 
 This is a simple text file.
-The end.
+Goodbye, world! </patch>
The following issue is:\n\n
"""
            + self.context.topic
        )  # problem statement
        # TODO potentially dont initiate CodeSimModel and instead just move the cosim function out and just import that
        self.codesim = CodeSimModel(code_scorer=code_scorer)

    def score(self, completion):
        try:
            completion = json.loads(completion)
        except:
            return 0
        
        if not completion:
            return 0
        
        total_points = len(self.diffs)  # one point per file
        points = total_points
        # check if the diff file names match the ones in the github issue
        for diff in self.diffs:
            if diff.file not in completion.keys():
                points -= 1

            miner_diff = parse_diff(completion[diff.file], no_title=True)[0]
            total_line_points = len(diff.edited_lines)
            line_points = 0
            for edit, line_num, content in diff.edited_lines:
                if line_num not in [line[1] for line in miner_diff.edited_lines]:
                    continue
                # find the miners edited line that has the same line number
                miner_edit, mine_line_num, miner_content = [
                    miner_line
                    for miner_line in miner_diff.edited_lines
                    if miner_line[1] == line_num
                ][0]
                if (
                    miner_edit == "-"
                    and miner_content == content
                    and miner_edit == edit
                ):
                    line_points += 1
                if miner_edit == "+" and miner_edit == edit:
                    line_points += 1 * self.codesim.similarity(content, miner_content)
            if line_points != 0:
                points += 1 * (line_points / total_line_points)
        if points == 0:
            return 0
        return points / total_points

    def reward(self, completions: List[str]) -> BatchRewardOutput:
        rewards = []
        timings = []

        for completion in completions:
            t0 = time.time()
            rewards.append(self.score(completion))
            timings.append(time.time() - t0)
        output = BatchRewardOutput(rewards=rewards, timings=timings, extra_info={})

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
