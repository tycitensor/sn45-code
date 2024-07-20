from .base import Dataset

from .github import GithubDataset
from .pip import PipDataset

DATASETS = {
    GithubDataset.name: GithubDataset(),
    PipDataset.name: PipDataset()    
}