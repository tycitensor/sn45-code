# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2024 Macrocosmos

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from datasets import load_dataset

from .base import Dataset
from coding.helpers.selector import Selector


class SWEFullDataset(Dataset):
    name = "swefull"

    def __init__(
        self,
    ):
        # load in princeton-nlp/SWE-bench and shuffle the dataset
        self.dataset = load_dataset("princeton-nlp/SWE-bench", split="train", streaming=True).shuffle()
        self.dataset_iterset = iter(self.dataset)

    def get(self, n=100, selector: Selector = Selector()) -> dict:
        row = next(self.dataset_iterset)
        return {
            "topic": row["problem_statement"],
            "title": row["repo"],
            "content": row["patch"],
            "extras": dict(pull_number="", base_commit=row["base_commit"]),
        }

    def search(self, query, selector: Selector = None, **kwargs):
        pass

    def random(self, n=100, selector: Selector = None, **kwargs):
        return self.get(n=100, selector=selector)
