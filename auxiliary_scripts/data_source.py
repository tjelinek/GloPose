from abc import ABC, abstractmethod


class DataProvider(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def next_image(self):
        pass

    @abstractmethod
    def next_segmentation(self):
        pass

