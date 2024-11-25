from .base import Dataset

from .bigcodebench import BigcodeBenchDataset
from .thestack import TheStackDataset
from .pip import PipDataset
from .swe import SWEDataset

class DatasetManager:
    def __init__(self, config = None):
        self._datasets = None
        self.config = config

    @property   
    def datasets(self):
        if self._datasets is None:
            self._datasets = {
                TheStackDataset.name: TheStackDataset(),
                PipDataset.name: PipDataset(),
                SWEDataset.name: SWEDataset(),
                BigcodeBenchDataset.name: BigcodeBenchDataset(self.config)
            }
        return self._datasets
