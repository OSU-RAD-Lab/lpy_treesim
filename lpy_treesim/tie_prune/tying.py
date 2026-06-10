"""
Defines type of tying
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np

from lpy_treesim.orchard_generation.generate_orchard import wire_asset


@dataclass
class WireBranchAttach:
    # where a branch is attached to a wire
    x_along: float = 0.0
    branch_id: int = -1
    attractor_pts: np.array = None
    attractor_dir: np.array = None
    spacing: float = 0.0


@dataclass
class TyingState:
    """Tying and guiding state for a wood object."""
    class TyingType(Enum):
        NO_TIE = "no_tie"
        TIE_ALONG = "tie_along"
        TIE_ALONG_FIRST = "tie_along_first"
        TIE_ACROSS = "tie_across"

    tie_needs_updating: bool = False  # Set to false when guide curve updated, true when branch changes/new guide point added
    wire_attach: WireBranchAttach = None  # These are the list of points to tie to
    last_tie_index: int = -1          # Branch has been tied to all of the points up to this index
    guide_points: list[tuple] = None  # Current set of guide points (control points of spline)
    tie_type: TyingType = TyingType.NO_TIE  # One of tying_type

    @property
    def is_tied(self):
        return self.wire_attach is not None

    def __post_init__(self):
        """Initialize guide_points as empty list if not provided."""
        if self.guide_points is None:
            self.guide_points = []




