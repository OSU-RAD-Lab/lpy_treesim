"""
Helper utilities for L-Py tree simulation system.

This module provides utility functions for procedural tree generation and simulation
in the L-Py framework. It includes functions for:
"""

import importlib


def resolve_attr(path: str):
    """Import a fully qualified attribute path."""
    pkg_path, attr_name = path.rsplit(".", 1)
    pkg = importlib.import_module(pkg_path)
    return getattr(pkg, attr_name)
