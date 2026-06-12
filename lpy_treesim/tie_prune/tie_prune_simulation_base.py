"""
Base module for adding tying and training to a trellis to an L-System tree

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
from scipy.optimize import linear_sum_assignment
from lpy_treesim.tie_prune.wire_support import Support
from lpy_treesim.lpy_functions.lpy_sring_prune_edit_fns import cut_from
from lpy_treesim.tie_prune.tying import TyingState
from lpy_treesim.lpy_functions.lpy_geometry_fns import create_bezier_curve
from lpy_treesim.tie_prune.tie_prune_configuration import SimulationConfig


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

        # Support Structure
        self.support = Support(start_height=config.start_height,
                               angle=config.angle,
                               spacing_wires=config.spacing_wires,
                               num_wires=config.num_wires,
                               x_left=config.x_left,
                               x_right=config.x_right)

        # Attractor grids will be added in generate_points
        self.trunk_attractor = None
        self.branch_attractor = None

        self.generate_attractor_grids()

    @abstractmethod
    def create_trunk_curve(self):
        """ Create an initial growth curve for the tree trunk. Defaults to straight up"""
        curve = create_bezier_curve(x_range = (-1, 1), y_range = (-1, 1), z_range = (0, 10), rng=self.config.lpy_rng)
        return curve

    @abstractmethod
    def generate_attractor_grids(self):
        """
        Generate 3D points for the trellis wire structure.

        This method must be implemented by architecture-specific subclasses
        to define the layout of trellis wires (V-trellis, UFO, etc.).

        Returns:
            list: List of (x, y, z) tuples representing wire attachment points
        """
        pass

    def get_energy_matrix(self, all_branches):
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

        Returns:
            numpy.ndarray: Energy matrix of shape (num_branches, num_wires) where
                          matrix[i][j] is the energy cost of assigning branch i to wire j.
                          Untied branches and occupied wires have infinite energy (np.inf).
        """
        branches = []
        for branch in all_branches:
            # Skip branches that are already tied
            if not branch.tying.is_tied:
                # And that haven't grown yet
                if branch.length > 0.0:
                    branches.append(branch)
        num_branches = len(branches)
        num_wires = len(self.branch_attractor)

        # Initialize energy matrix with infinite values (impossible assignments)
        energy_matrix = np.full((num_branches, num_wires), 10000.0)

        # Calculate energy costs for all valid branch-wire combinations
        for branch_idx, branch in enumerate(branches):
            if branch.tying.is_tied:
                continue

            for wire_id, wire in enumerate(self.branch_attractor):
                # Skip wires that already have a branch attached
                if wire.branch_id != -1:
                    continue

                # Calculate weighted distance energy for this branch-wire pair
                # Energy considers distance from wire to both branch endpoints
                wire_points = np.array(wire.attractor_pts)
                branch_start = np.array(branch.location.start)
                branch_end = np.array(branch.location.end)

                # Don't care about z just if it aligns (x) and is not too far from the wire (y)
                end_indx = 2
                if branch.tying.tie_type == TyingState.TyingType.TIE_ALONG:
                    # Care about z
                    end_indx = 3
                start_distance_energy = 1e30
                end_distance_energy = 1e30
                for row in range(0, wire_points.shape[0]):
                    start_energy = np.mean((wire_points[row, 0:end_indx] - branch_start[0:end_indx]) ** 2)
                    end_energy = np.mean((wire_points[row, 0:end_indx] - branch_end[0:end_indx]) ** 2)
                    if start_energy < start_distance_energy:
                        start_distance_energy = start_energy
                    if end_energy < end_distance_energy:
                        end_distance_energy = end_energy

                total_energy = (start_distance_energy + end_distance_energy) / 2.0

                energy_matrix[branch_idx, wire_id] = total_energy

        return energy_matrix, branches

    def decide_guide(self, energy_matrix, branches):
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

        Returns:
            None: Modifies branches and branch_attractors in-place with new assignments
        """
        num_branches, num_wires = energy_matrix.shape

        # Early return if no branches or wires to assign
        if num_branches == 0 or num_wires == 0:
            return

        # Run the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(energy_matrix)

        # Wires are organized from left to right (ufo) or up to down (envy);
        # Continue making assignments until no valid ones remain
        for branch_indx, wire_indx in zip(row_ind, col_ind):
            spacing = self.branch_attractor[wire_indx].spacing
            if energy_matrix[branch_indx, wire_indx] < spacing:
                # Get the branch and wire objects
                branch = branches[branch_indx]
                wire_attach = self.branch_attractor[wire_indx]

                # Perform the assignment
                branch.tying.wire_attach = wire_attach
                id = int(branch.name.split("_")[-1])
                wire_attach.branch_id = id

    def remove_children_from_hierarchy(self, branch_name, branch_hierarchy, parent_map):
        """
        Remove all children of a given branch from the hierarchy and parent map.

        This function is used during pruning to ensure that when a branch is pruned,
        all of its descendant branches are also removed from the branch hierarchy
        and the parent mapping. This prevents dangling references to pruned branches.

        Args:
            branch_name: Name of the branch whose children are to be removed
            branch_hierarchy: Dictionary mapping parent branch names to lists of child branches
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
                self.remove_children_from_hierarchy(child_branch.name, branch_hierarchy, parent_map)
            del branch_hierarchy[branch_name]
        
        # Remove branch and its children from parent_map
        if parent_map and branch_name in parent_map:
            del parent_map[branch_name]
    
    def prune(self, lstring, branch_hierarchy, parent_map=None):
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
                not_tied_to_wire = not branch.tying.is_tied
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
                        self.remove_children_from_hierarchy(branch.name, branch_hierarchy, parent_map)

                    return True

        return False

    def tie(self, lstring):
        """
        Perform tying operation on eligible branches in the L-System string.

        This function searches through the L-System string for 'WoodStart' modules that
        represent branches ready for tying to trellis wires. It identifies branches that:
        1. Have tying properties (tying attribute exists)
        2. Have a defined tie axis (tie_axis is not None)
        3. Have not been tied yet (tie_needs_updating is False)
        4. Have guide points available for wire attachment

        When an eligible branch is found, it performs the tying operation by:
        - Marking the branch as tied (tie_needs_updating = False)
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
        string_readable = str(lstring).replace("]", "]\n")
        print(f"Tying:\n{string_readable}\n")
        for position, symbol in enumerate(lstring):
            # Check if this is a WoodStart module with tying capabilities
            if symbol == "WoodStart":

                branch = symbol[0].type

                # lstring is a pointer, so this modifies the original as well
                lstring, modifications_count = branch.tie_lstring(lstring, position)
                if modifications_count > 0:
                    string_readable = str(lstring).replace("]", "]\n")
                    print(f"Tied{string_readable}\n\n")
                    return True

        return False
