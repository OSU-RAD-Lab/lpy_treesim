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
from openalea.lpy import Lsystem, newmodule

from scipy.optimize import linear_sum_assignment

from lpy_treesim.tie_prune.wire_support import Support
from lpy_treesim.lpy_functions.lpy_sring_prune_edit_fns import cut_using_string_manipulation
from lpy_treesim.tie_prune.tying import TyingState
from lpy_treesim.tree_models.base_tree.bud_site import BudSite
import numpy as np
from typing import Callable
from lpy_treesim.tie_prune.tie_prune_configuration import SimulationConfig
from lpy_treesim.tree_models.base_tree.tree_wood_prototypes import BasicWood


class TreeSimulationBase(ABC):
    """
    Base class for tree architecture simulations with trellis training.

    This class provides common algorithms for:
    - Energy-based optimization for branch-to-wire assignment
    - Assignment of branches to wires
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

        # These control when to stop letting buds turn into spurs/branches, and then generate geomety
        self.current_iteration: int = 0  # Set in start_common

        # These are set in the start iteration method
        self.end_bud_growth: bool = False
        self.end_growth: bool = False
        self.generate_geometry: bool = False  # Set to True when ready for lstring to have geom

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

    def start_iteration(self, lstring, branch_hierarchy: dict):
        """Shared pre-iteration tying preparation logic.
        @param lstring - the actual lstring being generated
        @param branch_hierarchy - the current branch hierarchy as a dictionary """

        if self.current_iteration >= self.config.derivation_length - 3:
            # First, freeze bud growth
            print("ENDING budding")
            self.end_bud_growth = True

        if self.current_iteration >= self.config.derivation_length - 2:
            # First, freeze bud growth
            print("ENDING growth")
            self.end_growth = True

        if self.current_iteration >= self.config.derivation_length - 1:
            # Simulation ending - generate the cylinders by replacing make_cylinder with _ F
            print("STARTING geometry")
            self.generate_geometry = True

        # If we haven't added the attractor for the main trunks, do so
        for indx, trunk in enumerate(branch_hierarchy["root"]):
            if not trunk.tying.wire_attach and len(self.trunk_attractor) > indx:
                trunk.tying.wire_attach = self.trunk_attractor[indx]

        return lstring

    def end_iteration(self,
                      branch_hierarchy : dict,
                      map_names_to_branches : dict,
                      get_iteration_number: Callable[[], int]):
        """Shared post-iteration tying and pruning orchestration."""
        sim_config = self.config

        if sim_config.do_trunk_tying(self.current_iteration):
            # Pin tree trunk one iteration before branches so vectors update correctly
            for trunk in branch_hierarchy["root"]:
                # Note: The bezier curve in the string will be updated the next time the lstring
                # is interpolated
                trunk.update_guide()

        trunk_branches = self.get_trunk_branches(branch_hierarchy=branch_hierarchy)

        if sim_config.do_branch_tying(self.current_iteration):
            # Estimate of cost to tie branches to open wire attachments
            energy_matrix, wire_ids, open_branches = self.get_energy_matrix(trunk_branches)

            # Actually tie some of the branches to the wires
            self.decide_guide(energy_matrix=energy_matrix, wire_ids=wire_ids, branches=open_branches)

            # Update guide curve - next time interpret lstring is called, it will use the new guide curve
            for branch in trunk_branches:
                branch.update_guide()

        if sim_config.do_pruning(self.current_iteration):
            self.prune(branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

        if sim_config.do_year_increment(self.current_iteration):
            for items in branch_hierarchy.values():
                for item in items:
                    item.add_year()

        self.current_iteration = get_iteration_number() + 1

    def get_trunk_branches(self, branch_hierarchy: dict) ->list[BasicWood]:
        """ Find all the buds that have branches growing from them"""
        branches = []
        for trunks in branch_hierarchy["root"]:
            for bud_site in branch_hierarchy[trunks.name]:
                if bud_site.branch_child:
                    branches.append(bud_site.branch_child)
        return branches

    def get_energy_matrix(self, branches: list[BasicWood]):
        """
        Calculate the energy matrix for optimal branch-to-wire assignment.

        This function computes an energy cost matrix where each entry represents the
        "cost" of assigning a specific branch to a specific wire in the trellis system.
        The energy is based on the Euclidean distance from the wire attachment point to
        the start of each branch and the angle between the start of the branch and the tie
        direction, weighted by the simulation's distance weight parameter.

        The algorithm uses a greedy optimization approach where branches are assigned
        to the lowest-energy available wire that hasn't reached capacity.

        Args:
            branches: List of branch objects to be assigned to wires

        Returns:
            numpy.ndarray: Energy matrix of shape (num_branches, num_wires) where
                          matrix[i][j] is the energy cost of assigning branch i to wire j.
                          Untied branches and occupied wires have infinite energy (np.inf).
        """
        open_branches = []
        for branch in branches:
            # Skip branches that are already tied
            if not branch.tying.is_tied:
                # And that haven't grown yet
                if branch.growth.length > 0.5 * self.support.spacing_wires:
                    open_branches.append(branch)
                else:
                    print(f"Skipping {branch.name}, too short {branch.growth.length}")

        wire_ids = []
        for wire_id, wire in enumerate(self.branch_attractor):
            # Skip wires that already have a branch attached
            if wire.branch_id == -1:
                wire_ids.append(wire_id)

        num_branches = len(open_branches)
        num_wires = len(wire_ids)

        # Initialize energy matrix with infinite values (impossible assignments)
        energy_matrix = np.full((num_branches, num_wires), 10000.0)

        # Calculate energy costs for all valid branch-wire combinations
        min_start_height = self.config.start_height - self.config.spacing_wires * 0.5
        min_dist = self.config.spacing_wires * 0.5
        print(f"Beginning energy matrix {num_branches} {num_wires}, start height {min_start_height} min dist {min_dist}")
        for branch_idx, branch in enumerate(open_branches):
            branch_start = np.array(branch.location.start)
            branch_dir = np.array(branch.location.start_dir)
            branch_end = np.array(branch.location.end)
            if branch_start[2] < min_start_height:
                print(f"Branch {branch.name} on trunk {branch_start} too far below wire ")
                continue

            for wire_idx, wire_id in enumerate(wire_ids):
                wire = self.branch_attractor[wire_id]

                # Calculate weighted distance energy for this branch-wire pair
                # Energy considers distance from wire to both branch endpoints
                wire_points = np.array(wire.attractor_pts)

                align = np.dot(branch_dir, wire.attractor_dir)
                if align < 0.0:
                    print(f"Branch {branch.name} dir {branch_dir}, wire attach dir {wire.attractor_dir}")
                    continue

                # Don't care about z just if it aligns (x) and is not too far from the wire (y)
                end_indx = 2
                if branch.tying.tie_type == TyingState.TyingType.TIE_ALONG:
                    # Care about z
                    end_indx = 3
                start_distance_energy = 1e30
                end_distance_energy = 1e30
                for row in range(0, wire_points.shape[0]):
                    start_energy = np.linalg.norm((wire_points[row, 0:end_indx] - branch_start[0:end_indx]) ** 2)
                    end_energy = np.linalg.norm((wire_points[row, 0:end_indx] - branch_end[0:end_indx]) ** 2)
                    if start_energy < start_distance_energy:
                        start_distance_energy = start_energy
                    if end_energy < end_distance_energy:
                        end_distance_energy = end_energy

                if start_distance_energy > self.support.spacing_wires / 2.0 and end_distance_energy > self.support.spacing_wires / 2.0:
                    print(f"Branch {branch.name} Too far away {start_distance_energy} {end_distance_energy}")
                    continue
                print(f"Branch {branch.name} Start {branch_start} end {branch_end}\nwire {wire_points}")
                dist_energy = 0.9 * start_distance_energy / self.support.spacing_wires + 0.1 * end_distance_energy / self.support.spacing_wires
                total_energy = self.config.energy_distance_weight * dist_energy + align * self.config.energy_angle_weight

                energy_matrix[branch_idx, wire_idx] = total_energy

        return energy_matrix, wire_ids, open_branches

    def decide_guide(self, energy_matrix, wire_ids: list, branches):
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
            wire_ids: Ids of wires that can be tied
            branches: List of branch objects to be assigned

        Returns:
            None: Modifies branches and branch_attractors in-place with new assignments
        """
        num_branches, num_wires = energy_matrix.shape

        # Early return if no branches or wires to assign
        if len(branches) == 0 or len(wire_ids) == 0:
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

    def prune(self, branch_hierarchy, map_names_to_branches):
        """
        Prune old branches that exceed the age_in_iterations threshold and haven't been tied to wires.

        This function implements the pruning strategy for the tree training simulation.
        It identifies branches that have grown too old (exceeding the pruning age_in_iterations threshold)
        but haven't been successfully tied to trellis wires.

        The pruning criteria are:
        1. Branch age_in_iterations exceeds the configured pruning threshold
        2. Branch has not been tied to any trellis wire
        3. Branch has not already been marked for cutting
        4. Branch is prunable (respects the prunable flag)

        When a branch meets all criteria, it is:
        - Removed from branch_hierarchy
        - Removed from parent's children list
        - Removed from the map names dictionary

        Note that the bud site that generated the pruned branch will remain and be marked pruned

        Args:
            branch_hierarchy: Dictionary mapping branch names to lists of child branches

        Returns:
            bool: True if a branch was pruned, False if no eligible branches found

        Note:
            This function processes one branch at a time and returns immediately after
            pruning a single branch. It should be called repeatedly (e.g., in a while loop)
            until no more pruning operations are possible. The cut_from() function handles
            the actual removal of the branch and any dependent substructures from the string.
        """

        # Collect names of buds/branches/spurs to be pruned
        buds_to_prune = []
        for branch_children in branch_hierarchy.values():
            for bud in branch_children:
                if not "bud" in bud.name:
                    continue
                if not bud.branch_child:
                    continue
                branch: BasicWood = bud.branch_child
                age_exceeds_threshold = branch.growth.age_in_iterations > self.config.pruning_age_threshold
                not_tied_to_wire = not branch.tying.is_tied
                is_prunable = branch.config.remove_at_age

                # Prune if all criteria are met
                if age_exceeds_threshold and not_tied_to_wire and is_prunable:
                    buds_to_prune.append(bud)

        # Now add the bud names to the list
        names_to_x = []
        for bud in buds_to_prune:
            names_to_x.extend(bud.prune())

        # Now remove the names from branch hierarchy
        for name in names_to_x:
            del branch_hierarchy[name]
            del map_names_to_branches[name]

        # In the next iteration WoodStart etc will be replaced with % and cut out
        print(f"Left: ")
        for key in map_names_to_branches.keys():
            if "rimary" in key and "bud" not in key:
                print(f"{key}")
