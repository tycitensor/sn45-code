import re
from pydantic import BaseModel
from typing import Callable, List, Dict

from .task import Task
from coding.helpers.git import GitRepo
from coding.schemas import Context, Patch, Edit


class PatchChunk(BaseModel):
    file_name: str
    start_index: int
    end_index: int
    content: str
    new_content: str


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
            edits.append(
                Edit(
                    file_name=current_file,
                    line_number=new_file_line_num,
                    line_content="",
                    new_line_content=line[1:].strip(),
                )
            )
            new_file_line_num += 1
        elif line.startswith("-") and not line.startswith("---"):
            # Line removed from old file
            edits.append(
                Edit(
                    file_name=current_file,
                    line_number=old_file_line_num,
                    line_content=line[1:].strip(),
                    new_line_content="",
                )
            )
            old_file_line_num += 1
        elif line.startswith(" "):
            # Context lines (lines present in both old and new files)
            old_file_line_num += 1
            new_file_line_num += 1

    return Patch(edits=edits)


# TODO ensure chunks within 2 lines of each other are grouped together
def chunk_patch(patch: Patch) -> List[PatchChunk]:
    chunks = []
    current_chunk = []
    current_file = None

    for i, edit in enumerate(patch.edits):
        # Start new chunk if file changes or line numbers not consecutive
        if not current_chunk:
            current_chunk.append(edit)
            current_file = edit.file_name
        elif (
            edit.file_name == current_file
            and edit.line_number == current_chunk[-1].line_number + 1
        ):
            current_chunk.append(edit)
        else:
            # Convert chunk to PatchChunk before appending
            if current_chunk:
                start_idx = current_chunk[0].line_number
                end_idx = current_chunk[-1].line_number
                content = "\n".join(edit.line_content for edit in current_chunk)
                new_content = "\n".join(edit.new_line_content for edit in current_chunk)
                chunks.append(PatchChunk(
                    file_name=current_file,
                    start_index=start_idx,
                    end_index=end_idx,
                    content=content,
                    new_content=new_content
                ))
            current_chunk = [edit]
            current_file = edit.file_name

        # Add final chunk
        if i == len(patch.edits) - 1:
            if current_chunk:
                start_idx = current_chunk[0].line_number
                end_idx = current_chunk[-1].line_number
                content = "\n".join(edit.line_content for edit in current_chunk)
                new_content = "\n".join(edit.new_line_content for edit in current_chunk)
                chunks.append(PatchChunk(
                    file_name=current_file,
                    start_index=start_idx,
                    end_index=end_idx,
                    content=content,
                    new_content=new_content
                ))

    return chunks

class SWEBenchTask(Task):
    name: str = "swebench"
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
        self.repo = GitRepo(context.title, context.extras["base_commit"])
        self.code_scorer = code_scorer
        self.context = context
        self.patch: Patch = parse_diff(context.content)
        self.query = context.topic
        # self.repo = context.title
        self.base_commit = context.extras["base_commit"]
        self.pull_number = context.extras["pull_number"]

    def score(self, patch: Patch, token_count: int):
        valid_num_lines = {}  # file name -> num lines
        miner_num_lines = {}

        for edit in self.patch.edits:
            if edit.file_name not in valid_num_lines:
                valid_num_lines[edit.file_name] = 0
            valid_num_lines[edit.file_name] += 1

            if edit.file_name not in miner_num_lines:
                miner_num_lines[edit.file_name] = 0
            miner_num_lines[edit.file_name] += 1

        # see which lines in valid patch are in miner patch and find percent
        # miner can edit extra lines but not less
        total_lines = 0
        lines_in_miner = 0
        for file_name in valid_num_lines:
            if file_name in miner_num_lines:
                valid_lines = [
                    edit.line_number
                    for edit in self.patch.edits
                    if edit.file_name == file_name
                ]
                miner_lines = [
                    edit.line_number
                    for edit in patch.edits
                    if edit.file_name == file_name
                ]
                lines_in_miner += len(set(valid_lines) & set(miner_lines))
                total_lines += len(valid_lines)
        percent_lines_in_miner = lines_in_miner / total_lines

        # Group edits into chunks by consecutive line numbers
        valid_chunks = chunk_patch(self.patch)
        miner_chunks = chunk_patch(patch)

        chunk_score = 0
        total_chunk_score = 0

        # find chunks that share an index in the same file
        for valid_chunk in valid_chunks:
            exists = False
            for miner_chunk in miner_chunks:
                if (
                    miner_chunk.file_name == valid_chunk.file_name
                    and abs(miner_chunk.start_index - valid_chunk.start_index) <= 2
                ):
                    miner_chunk_content = "\n".join(
                        [edit.line_content for edit in miner_chunk]
                    )
                    valid_chunk_content = "\n".join(
                        [edit.line_content for edit in valid_chunk]
                    )
                    # TODO this has to be a score between the original vs valid and the miner
                    chunk_score += self.code_scorer(
                        miner_chunk_content, valid_chunk_content
                    )
                    total_chunk_score += 1
                    exists = True
                    break
            if not exists:
                total_chunk_score += 1

        chunk_percent = chunk_score / total_chunk_score

        score = (5 * percent_lines_in_miner + 5 * chunk_percent) / 10

        return score
