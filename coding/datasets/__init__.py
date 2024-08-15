from .base import Dataset

from .github import GithubDataset
from .pip import PipDataset
from .swe import SWEDataset

class DatasetManager:
    def __init__(self):
        self._datasets = None

    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = {
                GithubDataset.name: GithubDataset(),
                PipDataset.name: PipDataset(),
                SWEDataset.name: SWEDataset()
            }
        return self._datasets

# Create a single instance of DatasetManager
DATASET_MANAGER = DatasetManager()