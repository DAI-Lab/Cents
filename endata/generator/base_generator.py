from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def train_model(self, dataset):
        pass

    @abstractmethod
    def generate(self, context_vars):
        pass

    @abstractmethod
    def save(self, path: str = None, epoch: int = None):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
