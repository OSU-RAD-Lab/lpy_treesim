"""
Helper utilities for L-Py tree simulation system.

This module contains methods for
- L-System string manipulation (cutting, pruning operations)

The functions in this module are used by various tree architecture implementations
(UFO, Envy, etc.) to perform branch pruning/cutting
"""

from openalea.plantgl.all import (
    NurbsCurve,
    Vector3,
    Vector4,
    Point4Array,
    Point2Array,
    Point3Array,
    Polyline2D,
    BezierCurve,
    BezierCurve2D,
)
from openalea.lpy import Lsystem, newmodule


def cut_from(pruning_position, lstring, lsystem_path=None):
    """
    Mark a position in the L-System string for cutting/pruning.

    Inserts a cut marker (%) after the specified pruning position in the
    L-System string. This marks the location where a branch should be
    removed during the pruning process.

    Args:
        pruning_position: Index in the L-System string where pruning should occur
        lstring: The L-System string to modify
        lsystem_path: Optional path to create a new L-System object (unused in current implementation)

    Returns:
        Modified L-System string with cut marker inserted
    """
    # Insert cut marker (%) before the [] containing WoodStart
    lstring.insertAt(pruning_position - 1, newmodule("%"))
    return lstring


def cut_using_string_manipulation(pruning_position, lstring):
    """
    Remove a complete branch segment from the L-System string.

    Cuts starting from the pruning position until the end of the branch segment,
    which is signified by a closing bracket ']'. Uses bracket balancing to handle
    nested branch structures correctly.

    Args:
        pruning_position: Starting index in the L-System string for the cut operation
        lstring: The L-System string to modify
        lsystem_path: Optional path to create a new L-System object with the modified string

    Returns:
        Modified L-System string with the branch segment removed, or a new L-System
        object if lsystem_path is provided
    """
    bracket_balance = 0
    current_position = pruning_position
    # This symbol should be the starting [
    search_position = pruning_position + 1
    total_length = len(lstring)

    # Traverse the string until we find the matching closing bracket
    while search_position < total_length:
        if lstring[current_position].name == "[":
            bracket_balance += 1
        elif lstring[current_position].name == "]":
            if bracket_balance == 0:
                # Found the matching closing bracket, stop here
                break
            else:
                bracket_balance -= 1

        # Remove the current element
        del lstring[current_position]
        search_position += 1

    # If a path is provided, create a new L-System object
    if lsystem_path is not None:
        new_lsystem = Lsystem(lsystem_path)
        new_lsystem.axiom = lstring
        return new_lsystem

    return lstring

