import time
import numpy as np
from typing import List
from .reward import (
    BaseRewardModel,
    BatchRewardOutput,
    RewardModelTypeEnum,
)
from coding.helpers.cosine import normalize_cosim


class CodeSimModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "codesim"

    def __init__(self, code_scorer=None, **kwargs):
        super().__init__()
        self.code_scorer = code_scorer

    def similarity(self, reference: str, completion: str) -> float:
        if not reference:
            return 0
        if not completion:
            return 0
        P, R, F1 = self.code_scorer.score([completion], [reference])
        score = F1.tolist()[0]
        return normalize_cosim(score)
    
    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        """
        Get the score between two strings.
        """

        rewards = []
        timings = []
        for completion in completions:
            t0 = time.time()
            rewards.append(self.similarity(reference, completion))
            timings.append(time.time() - t0)
        output = BatchRewardOutput(
            rewards=rewards,
            timings=timings,
            extra_info={}
        )

        return output
