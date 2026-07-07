#supress file print statements
print = lambda *args, **kwargs: None

"""
Base module for adding tying, training, and pruning to a trellis to an L-System tree

This module provides common functionality for tree architecture simulations including:
- Energy-based branch-to-wire optimization
- Actual tying code is in tying.py
- Pruning strategies for untied branches
- Pruning for length

Architecture-specific implementations (Envy, UFO, etc.) should inherit from this base
and implement architecture-specific methods for laying out wires and attachment points
"""

from abc import ABC, abstractmethod

from openalea.plantgl.scenegraph import BezierCurve
# For doing the wire-branch assignment
from scipy.optimize import linear_sum_assignment
# How the wires are arranged
from lpy_treesim.tie_prune.wire_support import Support
# Actual tying happens in tying.py
from lpy_treesim.tie_prune.tying import TyingState
import numpy as np
from typing import Callable
from lpy_treesim.tie_prune.tie_prune_configuration import SimulationConfig
from lpy_treesim.tree_models.base_tree.tree_wood_prototypes import BasicWood

from lpy_treesim.tie_prune.pruning_algo.prune_tree import prune_tree


class TreeSimulationBase(ABC):
    """
    Base class for tree architecture simulations with trellis training. See also SimulationConfig

    This class provides common algorithms for:
    - Energy-based optimization for branch-to-wire assignment
    - Assignment of branches to wires
    - Pruning operations for untied branches
    - Pruning for length
    - Note: Pruning happens by the branch elements being removed from the lpy string. This happens in base_lpy.lpy;
        if a branch is no longer in branch_hierarchy, then it will be removed from the string by placing a '%'
        in the string (see WoodStart, BudStart and SpurStart).
        So to prune a branch, remove it from both map_names_to_branches and branch_hierarchy
    - Note 2: Tying happens in tying.py by changing the guide curve, which is used in SetGuide. Because the lstring
        modules for interpretation are generated only to update/create the geometry, but not kept in the actual
        LString (they are re-generated whenever interpret string is called) we don't need to actually change the
        string. Instead, the next time SetGuide is called (see IStartBranch) it will grab the new guide curve
    - Note 3: If a branch is tied then the guide curve starts at the base of the curve and uses the global coordinate
        system (the @R symbol in IStartBranch). This just makes the math easier; otherwise, the guide curve is in the
        local coordinate system.

    Architecture-specific implementations should:
    1. Inherit from this class
    2. Implement generate_points() for their specific trellis layout (this turns the wires into sets of guide points)
    3. Optionally override methods if custom behavior for pruning is needed
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

        # Attractor points will be added in generate_points; one list for each trunk/branch that is tied down
        self.trunk_attractor = None
        self.branch_attractor = None

        # Override this method to set the tie points
        self.generate_attractor_grids()

        # This controls when to stop letting buds turn into spurs/branches, and then generate geometry
        self.current_iteration: int = 0  # Set in start_common

        # These are set in the start iteration method
        self.end_bud_growth: bool = False
        self.end_growth: bool = False
        self.generate_geometry: bool = False  # Set to True when ready for lstring to have geom

        # For energy guide
        self.invalid_attractor_value = 1000

    @abstractmethod
    def generate_attractor_grids(self):
        """
        Generate 3D points for the trellis wire structure.

        This method must be implemented by architecture-specific subclasses
        to define the layout of trellis wires (V-trellis, UFO, etc.).

        Sets the lists in:
            self.trunk_attractor
            self.branch_attractor
        """
        pass

    def start_iteration(self, lstring, branch_hierarchy: dict):
        """Shared pre-iteration tying preparation logic.
        Lstrings work by replacing module names with new modules. So creating a new branch happens
        by creating a bud site, then a bud site turns into a branch then the branch grows. This shuts down that
        process so that there are not dangling modules when geometry is finally created (the last iteration)
        derivation_length should be number of years times number of iterations per year (currently 28)
        This method is called at the start of every iteration (see base_lpy.lpy)
        @param lstring - the actual lstring being generated - not really used, but could be
        @param branch_hierarchy - the current branch hierarchy as a dictionary """

        if self.current_iteration >= self.config.derivation_length - 3:
            # First, freeze budding (do not generate any new bud sites)
            print("ENDING budding")
            self.end_bud_growth = True

        if self.current_iteration >= self.config.derivation_length - 2:
            # Next, freeze bud growth (no new branches/spurs from buds)
            print("ENDING growth")
            self.end_growth = True

        if self.current_iteration >= self.config.derivation_length - 1:
            # Simulation ending - generate the cylinders by replacing make_cylinder with _ F
            print("STARTING geometry")
            self.generate_geometry = True

        # If we haven't added the attractor for the main trunks, do so - this should happen on the first iteration
        for indx, trunk in enumerate(branch_hierarchy["root"]):
            if not trunk.tying.wire_attach and len(self.trunk_attractor) > indx:
                trunk.tying.wire_attach = self.trunk_attractor[indx]

        return lstring

    def end_iteration(self,
                      branch_hierarchy: dict,
                      map_names_to_branches: dict,
                      get_iteration_number: Callable[[], int]):
        """This is where the actual pruning and tying gets called. Reminder that the next iteration will result in the
           change to the actual string:
           1) Any items removed from branch_hierarchy and map_names_to branches will be x'd out. Reminder to remove
           from BOTH data structures
           2) Any new guide curves will be used the next time intepret string is called (see generate_tree in tree_builder_lpy.py)
            This method is called at the end of every iteration (see base_lpy.lpy)
           """
        # The saved SimulationConfig data
        sim_config = self.config

        if sim_config.do_trunk_tying(self.current_iteration):
            # Pin tree trunk one iteration before branches so heading vectors for branches update correctly
            # Note that the trunk is pinned on the first iteration (see bottom of start_iteration)
            for trunk in branch_hierarchy["root"]:
                # Note: The bezier curve in the string (used in SetGuide, see IMakeCylinder in base_lpy.lpy) will be
                # updated the next time the lstring is interpolated
                trunk.update_guide()

        trunk_branches = self.get_trunk_branches(branch_hierarchy=branch_hierarchy)

        # This happens at the end of every year; pick branches to tie to the wires
        if sim_config.do_branch_tying(self.current_iteration):
            trunk = branch_hierarchy["root"][0]
            #  This will correctly map the dist_along parameter in the bud to the actual point on the curve
            """
            crv: BezierCurve = trunk.growth_curve
            u_to_len_map = crv.getArcLengthToUMapping()
            for bud in trunk.bud_sites:
                t = bud.dist_along / crv.getLength()
                t_arc_length = u_to_len_map(t)
                loc = crv.getPointAt(t_arc_length)
                print(f" {bud.start_loc} loc {loc}", end="")
            """

            # Estimate of cost to tie branches to open wire attachments
            energy_matrix, wire_ids, open_branches = self.get_energy_matrix(trunk_branches)

            # Actually tie some of the branches to the wires
            self.decide_guide(energy_matrix=energy_matrix, wire_ids=wire_ids, branches=open_branches)

            # Update guide curve - next time interpret lstring is called, it will use the new guide curve
            for branch in trunk_branches:
                branch.update_guide()

        # This happens at the end of every year one iteration after the branches are tied and (optionally) for
        #   summer pruning
        if sim_config.do_pruning(self.current_iteration):
            # proceed to prune the tree
            prune_tree(self, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

        # The branches track what year they are so that growth rates can change per year
        if sim_config.do_year_increment(self.current_iteration):
            for items in branch_hierarchy.values():
                for item in items:
                    item.add_year()

        self.current_iteration = get_iteration_number() + 1

    @staticmethod
    def get_trunk_branches(branch_hierarchy: dict) -> list[BasicWood]:
        """ Find all the buds on the trunk that have branches growing from them"""
        branches = []
        for trunks in branch_hierarchy["root"]:
            for bud_site in branch_hierarchy[trunks.name]:
                if bud_site.branch_child:
                    branches.append(bud_site.branch_child)
        return branches

    def get_energy_matrix(self, branches: list[BasicWood]) -> (np.array, list[int], list[BasicWood]):
        """
        Calculate the energy matrix for optimal branch-to-wire assignment.

        This function computes an energy cost matrix where each entry represents the
        "cost" of assigning a specific branch to a specific wire/set of wires in the trellis system.
        The energy is based on the Euclidean distance from the wire attachment point to
        the start of each branch and the angle between the start of the branch and the tie
        direction, weighted by the simulation's distance weight parameter.

        Note: Skips branches that are too short/pointing the wrong way

        Args:
            branches: List of branch objects to be assigned to wires

        Returns:
            numpy.ndarray: Energy matrix of shape (num_branches, num_wires) where
                          matrix[i][j] is the energy cost of assigning branch i to wire j.
                          Untied branches and occupied wires have infinite energy (np.inf).
            list[int]: List of wire ids used (wires that already have branches assigned are skipped)
            list[BasicWood]: Available branches
        """
        open_branches = []
        for branch in branches:
            # Skip branches that are already tied
            if not branch.tying.is_tied:
                # And that haven't grown yet
                if branch.growth.length > 0.5 * self.support.spacing_wires:
                    open_branches.append(branch)
                else:
                    print(f"Skipping {branch.name}, {branch.location.start} too short {branch.growth.length}")

        wire_ids = []
        for wire_id, wire in enumerate(self.branch_attractor):
            # Skip wires that already have a branch attached
            if wire.branch_id == -1:
                wire_ids.append(wire_id)

        num_branches = len(open_branches)
        num_wires = len(wire_ids)

        # Initialize energy matrix with infinite values (impossible assignments)
        energy_matrix = np.full((num_branches, num_wires), self.invalid_attractor_value)

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

            spacing = self.support.spacing_wires
            if branch.tying.tie_type == TyingState.TyingType.TIE_ACROSS:
                spacing = self.support.spacing_across_wire()

            for wire_idx, wire_id in enumerate(wire_ids):
                wire = self.branch_attractor[wire_id]

                # Calculate weighted distance energy for this branch-wire pair
                # Energy considers distance from wire to both branch endpoints
                wire_points = np.array(wire.attractor_pts)

                vec_to_wire_pt = wire_points[0, 0:3] - branch_start
                len_vec_to_wire = np.linalg.norm(vec_to_wire_pt)
                if not np.isclose(len_vec_to_wire, 0.0):
                    vec_to_wire_pt /= len_vec_to_wire
                    align_growth = np.dot(branch_dir, vec_to_wire_pt)
                    if align_growth < 0.45:
                        print(f"Branch {branch.name} dir {branch_dir}, wire attach dir {vec_to_wire_pt} wrong way")
                        continue

                align = np.dot(branch_dir, wire.attractor_dir)
                if align < 0.0:
                    # Skip branches that are currently pointing away from the tie direction
                    print(f"Branch {branch.name} dir {branch_dir}, wire attach dir {wire.attractor_dir}")
                    continue

                end_indx = 3
                if branch.tying.tie_type == TyingState.TyingType.TIE_ACROSS:
                    # Don't care about z just if it aligns (x) and is not too far from the wire (y)
                    end_indx = 2

                # Find the closest wire attachment point; for the start point it's probably the first wire point,
                #   for the end point it may be one further along
                # End point distance is an approximate measure of if the branch is growing in the direction of the wire
                start_distance_energy = 1e30
                end_distance_energy = 1e30
                for row in range(0, wire_points.shape[0]):
                    start_energy = 0.0
                    end_energy = 0.0
                    for icoord in range(0, end_indx):
                        start_energy += np.fabs(branch_start[icoord] - wire_points[row, icoord])
                        end_energy += np.fabs(branch_end[icoord] - wire_points[row, icoord])
                    if start_energy < start_distance_energy:
                        start_distance_energy = start_energy
                    if end_energy < end_distance_energy:
                        end_distance_energy = end_energy

                start_distance_energy /= end_indx
                end_distance_energy /= end_indx
                if start_distance_energy > spacing / 2.0 and end_distance_energy > spacing:
                    print(f"Branch {branch.name} Too far away {start_distance_energy} {end_distance_energy}")
                    continue
                print(f"Branch {branch.name} Start {branch_start} end {branch_end}\n          wire {wire_points[0, :]} spacing {spacing}")
                print(f" Energy {start_distance_energy:0.3f} {end_distance_energy:0.3f}", end="")
                # Weight the starting distance energy as more important than the ending one
                dist_energy = 0.8 * start_distance_energy / spacing + 0.2 * end_distance_energy / spacing
                print(f" Dist {dist_energy}", end="")
                # Weight the distance versus the angle
                #   Distance is scaled 0..1 based on wire spacing, align is dot product 0 to 1. 1 is better
                total_energy = self.config.energy_distance_weight * dist_energy + align * self.config.energy_angle_weight
                print(f" Total {total_energy}")

                energy_matrix[branch_idx, wire_idx] = total_energy

        return energy_matrix, wire_ids, open_branches

    def decide_guide(self, energy_matrix, wire_ids: list, branches):
        """
        Perform assignment of branches to wires based on energy matrix.

        This function uses the linear_sum_assignment method in scipy to find teh optimal minimum energy assignment.
        Once a branch is assigned to a wire, both that branch and wire are marked as unavailable to prevent further assignments.

        Note that this is not a 1-1, onto assignment - it only uses 'reasonable' pairings (see energy matrix above)
        Args:
            energy_matrix: numpy.ndarray of shape (num_branches, num_wires) with energy costs
            wire_ids: Ids of wires that can be tied
            branches: List of branch objects to be assigned

        Returns:
            None: Modifies branches and wire_attractors in-place with new assignments
        """
        num_branches, num_wires = energy_matrix.shape

        # Early return if no branches or wires to assign
        if num_branches == 0 or num_wires == 0:
            return

        # Run the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(energy_matrix)
        print(f"Energy matrix {energy_matrix}")

        # Wires are organized from left to right (ufo) or up to down (envy);
        # Continue making assignments as long as energy matrix < 1000

        for branch_indx, wire_indx in zip(row_ind, col_ind):
            if energy_matrix[branch_indx, wire_indx] < self.invalid_attractor_value:
                print(f" Assigning branch {branches[branch_indx].name} to wire {self.branch_attractor[wire_indx].attractor_pts[0]} {energy_matrix[branch_indx, wire_indx]}")
                # Get the branch and wire objects
                branch = branches[branch_indx]
                wire_id = wire_ids[wire_indx]
                wire_attach = self.branch_attractor[wire_id]

                # Perform the assignment
                branch.tying.wire_attach = wire_attach
                branch_id = int(branch.name.split("_")[-1])
                wire_attach.branch_id = branch_id