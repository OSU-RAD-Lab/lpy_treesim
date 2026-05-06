"""
Base simulation module for L-System tree training and trellis systems.

This module provides common functionality for tree architecture simulations including:
- Energy-based branch-to-wire optimization
- Tying operations for attaching branches to trellis wires
- Pruning strategies for untied branches

Architecture-specific implementations (Envy, UFO, etc.) should inherit from this base
and implement architecture-specific methods like point generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from lpy_treesim.helper import cut_from


@dataclass
class SimulationConfig(ABC):
    """Base configuration class for tree training simulations.

    Architecture-specific configs should inherit from this and add their own parameters.
    Common parameters across all architectures are defined here.

    Note: Labeling options (semantic_label, instance_label, per_cylinder_label) are
    configured via command-line arguments in make_n_trees.py and passed as extern
    variables to the L-system.
    """

    # Tying and Pruning Intervals
    num_iteration_tie: int = 5
    num_iteration_prune: int = 16

    # Support Structure
    support_num_wires: int = 14
    support_spacing_wires: int = 1
    support_trunk_wire_point: tuple = None

    # Energy and Tying Parameters
    energy_distance_weight: float = 0.5  # Weight for distance in energy calculation
    energy_threshold: float = 1.0  # Maximum energy threshold for tying

    # Pruning Parameters
    pruning_age_threshold: int = 6  # Age threshold for pruning untied branches

    # L-System Parameters
    derivation_length: int = 256  # Number of derivation steps
    use_generalized_cylinder: bool = False  # Whether to wrap new branches in @Gc/@Ge blocks

    # Growth Parameters
    tolerance: float = 1e-5  # Tolerance for comparison between floats

    # Visualization Parameters
    attractor_point_width: int = 10  # Width of attractor points in visualization


class TreeSimulationBase(ABC):
    """
    Base class for tree architecture simulations with trellis training.

    This class provides common algorithms for:
    - Energy-based optimization for branch-to-wire assignment
    - Greedy assignment of branches to wires
    - Pruning operations for untied branches
    - Tying operations to modify L-System strings

    Architecture-specific implementations should:
    1. Inherit from this class
    2. Implement generate_points() for their specific trellis layout
    3. Optionally override methods if custom behavior is needed
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation with a configuration object.

        Args:
            config: SimulationConfig instance with parameters for the simulation
        """
        self.config = config

    @abstractmethod
    def generate_points(self):
        """
        Generate 3D points for the trellis wire structure.

        This method must be implemented by architecture-specific subclasses
        to define the layout of trellis wires (V-trellis, UFO, etc.).

        Returns:
            list: List of (x, y, z) tuples representing wire attachment points
        """
        pass

    def get_energy_mat(self, branches, arch):
        """
        Calculate the energy matrix for optimal branch-to-wire assignment.

        This function computes an energy cost matrix where each entry represents the
        "cost" of assigning a specific branch to a specific wire in the trellis system.
        The energy is based on the Euclidean distance from wire attachment points to
        both the start and end points of each branch, weighted by the simulation's
        distance weight parameter.

        The algorithm uses a greedy optimization approach where branches are assigned
        to the lowest-energy available wire that hasn't reached capacity.

        Args:
            branches: List of branch objects to be assigned to wires
            arch: Support architecture object containing wire information

        Returns:
            numpy.ndarray: Energy matrix of shape (num_branches, num_wires) where
                          matrix[i][j] is the energy cost of assigning branch i to wire j.
                          Untied branches and occupied wires have infinite energy (np.inf).
        """
        num_branches = len(branches)
        num_wires = len(arch.branch_supports)

        # Initialize energy matrix with infinite values (impossible assignments)
        energy_matrix = np.full((num_branches, num_wires), np.inf)

        # Calculate energy costs for all valid branch-wire combinations
        for branch_idx, branch in enumerate(branches):
            # Skip branches that are already tied
            if branch.tying.has_tied:
                continue

            for wire_id, wire in arch.branch_supports.items():
                # Skip wires that already have a branch attached
                if wire.num_branch >= 1:
                    continue

                # Calculate weighted distance energy for this branch-wire pair
                # Energy considers distance from wire to both branch endpoints
                wire_point = np.array(wire.point)
                branch_start = np.array(branch.location.start)
                branch_end = np.array(branch.location.end)

                start_distance_energy = np.sum((wire_point - branch_start) ** 2)
                end_distance_energy = np.sum((wire_point - branch_end) ** 2)

                total_energy = (start_distance_energy + end_distance_energy) * self.config.energy_distance_weight

                energy_matrix[branch_idx, wire_id] = total_energy

        return energy_matrix

    def decide_guide(self, energy_matrix, branches, arch):
        """
        Perform greedy assignment of branches to wires based on energy matrix.

        This function implements a greedy optimization algorithm that iteratively assigns
        the branch-wire pair with the lowest energy cost. Once a branch is assigned to
        a wire, both that branch and wire are marked as unavailable (infinite energy)
        to prevent further assignments.

        The algorithm continues until no valid assignments remain (all remaining energies
        are infinite or above the threshold).

        Args:
            energy_matrix: numpy.ndarray of shape (num_branches, num_wires) with energy costs
            branches: List of branch objects to be assigned
            arch: Support architecture containing wire information

        Returns:
            None: Modifies branches and arch in-place with new assignments
        """
        num_branches, num_wires = energy_matrix.shape

        # Early return if no branches or wires to assign
        if num_branches == 0 or num_wires == 0:
            return

        # Continue making assignments until no valid ones remain
        while True:
            # Find the minimum energy value and its position
            min_energy_indices = np.argwhere(energy_matrix == np.min(energy_matrix))

            # If no valid indices found or matrix is empty, stop
            if len(min_energy_indices) == 0:
                break

            # Get the first (and typically only) minimum energy position
            branch_idx, wire_id = min_energy_indices[0]
            min_energy = energy_matrix[branch_idx, wire_id]

            # Stop if minimum energy is infinite (no valid assignments) or above threshold
            if np.isinf(min_energy) or min_energy > self.config.energy_threshold:
                break

            # Get the branch and wire objects
            branch = branches[branch_idx]
            wire = arch.branch_supports[wire_id]

            # Skip if branch is already tied (defensive check)
            if branch.tying.has_tied:
                # Mark this assignment as invalid and continue
                energy_matrix[branch_idx, wire_id] = np.inf
                continue

            # Perform the assignment
            branch.tying.guide_target = wire
            wire.add_branch()

            # Mark branch and wire as unavailable for future assignments
            # Set entire row (branch) to infinity - this branch can't be assigned again
            energy_matrix[branch_idx, :] = np.inf
            # Set entire column (wire) to infinity - this wire can't accept more branches
            energy_matrix[:, wire_id] = np.inf


    def remove_children_from_hierarchy(self, branch_name, branch_hierarchy, color_manager, parent_map):
        """
        Remove all children of a given branch from the hierarchy and parent map.

        This function is used during pruning to ensure that when a branch is pruned,
        all of its descendant branches are also removed from the branch hierarchy
        and the parent mapping. This prevents dangling references to pruned branches.

        Args:
            branch_name: Name of the branch whose children are to be removed
            branch_hierarchy: Dictionary mapping parent branch names to lists of child branches
            color_manager: ColorManager instance to remove color assignments
            parent_map: Dictionary mapping child branch names to their parent branch names
        Returns:
            None: Modifies branch_hierarchy and parent_map in-place
        """
        parent_name = parent_map.get(branch_name)
        
        if parent_name and parent_name in branch_hierarchy:
            # Remove the branch from its parent's children list
            children = branch_hierarchy[parent_name]
            branch_hierarchy[parent_name] = [child for child in children if child.name != branch_name]
        
        # Recursively remove all children of the branch first
        if branch_name in branch_hierarchy:
            for child_branch in branch_hierarchy[branch_name]:
                self.remove_children_from_hierarchy(child_branch.name, branch_hierarchy, color_manager, parent_map)
            del branch_hierarchy[branch_name]
        
        # Remove branch and its children from parent_map
        if parent_map and branch_name in parent_map:
            del parent_map[branch_name]

    
    def prune(self, lstring, branch_hierarchy, color_manager, parent_map=None):
        """
        Prune old branches that exceed the age threshold and haven't been tied to wires.

        This function implements the pruning strategy for the tree training simulation.
        It identifies branches that have grown too old (exceeding the pruning age threshold)
        but haven't been successfully tied to trellis wires. Such branches are considered
        unproductive and are removed from the L-System to encourage new growth.

        The pruning criteria are:
        1. Branch age exceeds the configured pruning threshold
        2. Branch has not been tied to any trellis wire
        3. Branch has not already been marked for cutting
        4. Branch is prunable (respects the prunable flag)

        When a branch meets all criteria, it is:
        - Marked as cut (to prevent re-processing)
        - Removed from the L-System string using cut_from()
        - Removed from parent's children list (if parent_map provided)

        Args:
            lstring: The current L-System string containing modules and their parameters
            branch_hierarchy: Dictionary mapping branch names to lists of child branches
            parent_map: Optional dictionary mapping child branch names to parent branch names
                       for efficient removal from parent's children list

        Returns:
            bool: True if a branch was pruned, False if no eligible branches found

        Note:
            This function processes one branch at a time and returns immediately after
            pruning a single branch. It should be called repeatedly (e.g., in a while loop)
            until no more pruning operations are possible. The cut_from() function handles
            the actual removal of the branch and any dependent substructures from the string.
        """
        for position, symbol in enumerate(lstring):
            # Check if this is a WoodStart module (represents a branch)
            if symbol.name == "WoodStart":
                branch = symbol[0].type

                # Check pruning criteria
                age_exceeds_threshold = branch.info.age > self.config.pruning_age_threshold
                not_tied_to_wire = not branch.tying.has_tied or branch.tying.guide_target is None
                not_already_cut = not branch.info.cut
                is_prunable = branch.info.prunable

                # Prune if all criteria are met
                if age_exceeds_threshold and not_tied_to_wire and not_already_cut and is_prunable:
                    # Mark branch as cut to prevent re-processing
                    branch.info.cut = True

                    # Remove the branch from the L-System string
                    lstring = cut_from(position, lstring)
                    print(f"Pruning {branch.name}")
                    
                    # Remove branch and its children from hierarchy and color manager
                    if parent_map is not None:
                        self.remove_children_from_hierarchy(branch.name, branch_hierarchy, color_manager, parent_map)

                    return True

        return False

    def tie(self, lstring):
        """
        Perform tying operation on eligible branches in the L-System string.

        This function searches through the L-System string for 'WoodStart' modules that
        represent branches ready for tying to trellis wires. It identifies branches that:
        1. Have tying properties (tying attribute exists)
        2. Have a defined tie axis (tie_axis is not None)
        3. Have not been tied yet (tie_updated is False)
        4. Have guide points available for wire attachment

        When an eligible branch is found, it performs the tying operation by:
        - Marking the branch as tied (tie_updated = False)
        - Adding the branch to the target wire
        - Calling the branch's tie_lstring method to modify the L-System string

        Args:
            lstring: The current L-System string containing modules and their parameters

        Returns:
            bool: True if a tying operation was performed, False if no eligible branches found

        Note:
            This function processes one branch at a time and returns immediately after
            tying a single branch. It should be called repeatedly (e.g., in a while loop)
            until no more tying operations are possible.
        """
        for position, symbol in enumerate(lstring):
            # Check if this is a WoodStart module with tying capabilities
            if (
                symbol == "WoodStart"
                and hasattr(symbol[0].type, "tying")
                and getattr(symbol[0].type.tying, "tie_axis", None) is not None
            ):

                branch = symbol[0].type

                # Skip branches that have already been processed for tying
                if not branch.tying.tie_updated:
                    continue

                # Check if branch has guide points for wire attachment
                if branch.tying.guide_points:
                    # Perform the tying operation
                    branch.tying.tie_updated = False

                    # Update the L-System string with tying modifications
                    lstring, modifications_count = branch.tie_lstring(lstring, position)

                    return True

        return False
