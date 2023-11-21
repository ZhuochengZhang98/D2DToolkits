import os
import sys
import importlib

from .document_config import DocumentConfig
from .document_dataset import DocumentDataset
from .document_translation_task import DocumentTranslation


__all__ = [
    "DocumentTranslation",
    "DocumentConfig",
    "DocumentDataset",
]


# import all examples
example_path = os.path.join(os.path.dirname(__file__), "..", "examples")
sys.path.insert(0, example_path)
for example_module in os.listdir(example_path):
    if os.path.isdir(os.path.join(example_path, example_module)):
        try:
            importlib.import_module(example_module)
            __all__.append(example_module)
        except ModuleNotFoundError:
            pass
