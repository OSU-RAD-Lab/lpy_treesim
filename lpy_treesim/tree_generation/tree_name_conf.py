#!/usr/bin/env python3
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreeNamingConfig:
    MAX_TREES = 99_999

    def __init__(self, namespace: str, tree_type: str):
        self.namespace = namespace.lower().strip()
        self.tree_type = tree_type.lower().strip()
        return

    def _prefix(self, index: int) -> str:
        if index > TreeNamingConfig.MAX_TREES:
            logging.error(f"Tree index {index} exceeds maximum supported value {TreeNamingConfig.MAX_TREES}.")
            raise ValueError(f"Tree index {index} exceeds maximum supported value {TreeNamingConfig.MAX_TREES}.")
        return f"{self.namespace}_{self.tree_type}_{index:05d}"

    def mesh_filename(self, index: int, file_type:str) -> str:
        """ File type is one of obj, ply, etc - something trimesh understands"""
        return f"{self._prefix(index)}.{file_type}"

    def usd_filename(self, index: int) -> str:
        return f"models/{self._prefix(index)}.usda"

    def color_map_filename(self, index: int) -> str:
        return f"{self._prefix(index)}_metadata.json"

    def hierarchy_filename(self, index: int) -> str:
        return f"{self._prefix(index)}_hierarchy.json"

    def metadata_filename(self, index: int) -> str:
        return f"{self._prefix(index)}_metadata.json"
