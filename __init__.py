# Similar to
# https://github.com/charlesfrye/twitter95/blob/main/backend/common/__init__.py

import modal

from model import AttentionModel, Dataset

mount = modal.Mount.from_local_python_packages(".")

__all__ = ["AttentionModel", "Dataset", "mount"]
