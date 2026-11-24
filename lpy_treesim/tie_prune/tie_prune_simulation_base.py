#supress file print statements
#print = lambda *args, **kwargs: None

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
from lpy_treesim.tie_prune.pruning_algo.ltr_and_data_structure import NeighborsDataStructure, LtrHuristic

from pandas import DataFrame as DF
from pathlib import Path
import copy


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
        self.generate_attractor_grids(self.config)

        # This controls when to stop letting buds turn into spurs/branches, and then generate geometry
        self.current_iteration: int = 0  # Set in start_common
        self.snapshot_iteration: int = 0
        self.freeze_for_snapshot = False

        # These are set in the start iteration method
        self.end_bud_growth: bool = False
        self.end_growth: bool = False
        self.generate_geometry: bool = False  # Set to True when ready for lstring to have geom
        self.mark: bool = True # Set to True when create a snapshot tree (will mark without making pruning cuts)
        # For energy guide
        self.invalid_attractor_value = 1000

    @abstractmethod
    def generate_attractor_grids(self, config: SimulationConfig):
        """
        Generate 3D points for the trellis wire structure.

        This method must be implemented by architecture-specific subclasses
        to define the layout of trellis wires (V-trellis, UFO, etc.).

        Sets the lists in:
            self.trunk_attractor
            self.branch_attractor
        """
        pass

    def start_iteration(self, 
                        lstring, 
                        branch_hierarchy: dict,
                        map_names_to_branches: dict, 
                        get_iteration_number: Callable[[], int]):
        """Shared pre-iteration tying preparation logic.
        Lstrings work by replacing module names with new modules. So creating a new branch happens
        by creating a bud site, then a bud site turns into a branch then the branch grows. This shuts down that
        process so that there are not dangling modules when geometry is finally created (the last iteration)
        derivation_length should be number of years times number of iterations per year (currently 28)
        This method is called at the start of every iteration (see base_lpy.lpy)
        @param lstring - the actual lstring being generated - not really used, but could be
        @param branch_hierarchy - the current branch hierarchy as a dictionary """

        self.current_iteration = get_iteration_number()
        
        # print(f'Base Iteration {self.current_iteration} Starting')
        #print(f'Base snapshot_iteration {self.snapshot_iteration}')

        if self.config.get_snapshot(self.current_iteration) and self.freeze_for_snapshot == False:
            print("deepcopying branch_hierarchy and map_names_to_branches")
            self.map_copy = copy.deepcopy(map_names_to_branches)
            self.hierarchy_copy = copy.deepcopy(branch_hierarchy)
            self.freeze_for_snapshot = True
            
        if self.freeze_for_snapshot:
                if self.snapshot_iteration == 5:
                    print("RESET to continue")
                    self.end_bud_growth = False
                    self.end_growth = False
                    self.generate_geometry = False
                    self.freeze_for_snapshot = False
                    self.mark = False
                    
                    # 1. Turn pruning back on so the script runs in Years 2-6
                    self.prune = True
                    
                    # 2. Safely restore BOTH the bud states AND the probabilities from the backup.
                    for name, original_obj in map_names_to_branches.items():
                        if name in self.map_copy:
                            if hasattr(original_obj, 'bud_state'):
                                original_obj.bud_state = self.map_copy[name].bud_state
                            if hasattr(original_obj, 'bud_break_probabilities'):
                                original_obj.bud_break_probabilities = self.map_copy[name].bud_break_probabilities
                            
                    # 3. Free up memory safely
                    del self.map_copy
                    del self.hierarchy_copy
                    
                    self.snapshot_iteration = 0

                else:
                    if self.snapshot_iteration >= 0:
                        # First, freeze budding (do not generate any new bud sites)
                        print("ENDING budding")
                        self.end_bud_growth = True
                        
                    #if self.snapshot_iteration >= 3:
                        # Next, freeze bud growth (no new branches/spurs from buds)
                        #print("ENDING growth")
                        #self.end_growth = True

                    if self.snapshot_iteration >= 4:
                        # Simulation ending - generate the cylinders by replacing make_cylinder with _ F
                        print("STARTING geometry")
                        self.generate_geometry = True
                        

                    self.snapshot_iteration += 1

        else:
            # TODO: still getting no mesh errors for final tree generation
            if self.current_iteration >= self.config.derivation_length - 3:
                # First, freeze budding (do not generate any new bud sites)
                print("Final ENDING budding")
                self.end_bud_growth = True

            if self.current_iteration >= self.config.derivation_length - 2:
                # Next, freeze bud growth (no new branches/spurs from buds)
                print("Final ENDING growth")
                self.end_growth = True

            if self.current_iteration >= self.config.derivation_length - 1:
                # Simulation ending - generate the cylinders by replacing make_cylinder with _ F
                print("Final STARTING geometry")
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
            print('TYING TRUNK')
            # Pin tree trunk one iteration before branches so heading vectors for branches update correctly
            # Note that the trunk is pinned on the first iteration (see bottom of start_iteration)
            for trunk in branch_hierarchy["root"]:
                # Note: The bezier curve in the string (used in SetGuide, see IMakeCylinder in base_lpy.lpy) will be
                # updated the next time the lstring is interpolated
                trunk.update_guide()

        trunk_branches = self.get_trunk_branches(branch_hierarchy=branch_hierarchy)

        # This happens at the end of every year; pick branches to tie to the wires
        if sim_config.do_branch_tying(self.current_iteration):
            print('TYING BRANCHES')
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

            if self.mark:
                # Pruning that happens every year (currently set to every 28 iterations

                # ltr should run first
                print('RUNNING MARKING LTR')
                ltr = LtrHuristic(map_names_to_branches=map_names_to_branches, tree_sim=self)
                
                # 1. Get TCSA baseline
                tcsa = ltr.get_tcsa(height_m=0.3)
                
                # 2. Get the list of primary limbs
                primary_limbs = ltr.get_primary_limbs()
                
                # 3. Calculate LCSA metrics
                limb_metrics = ltr.get_lcsa_metrics(primary_limbs=primary_limbs, measurement_dist_m=0.025)
                
                # 4. Run the simulated LTR logic (purely mathematical, no pruning)
                ltr_results = ltr.simulate_ltr_pruning(tcsa_cm2=tcsa, limb_metrics=limb_metrics, target_ltr=0.5)

                
                # Standard yearly structural pruning
                print("MARKING CUTS")
                prune_tree(self,
                            branch_hierarchy=branch_hierarchy, 
                            map_names_to_branches=map_names_to_branches,
                            mark = True)


            else:
                print('PRUNING TREE WITHOUT MARKING')
                prune_tree(self,
                           branch_hierarchy=branch_hierarchy, 
                           map_names_to_branches=map_names_to_branches,
                           mark = False)
                self.mark = True
                
                
            # The branches track what year they are so that growth rates can change per year
            if sim_config.do_year_increment(self.current_iteration):
                for items in branch_hierarchy.values():
                    for item in items:
                        item.add_year()

        for name, obj in map_names_to_branches.items():
            if hasattr(obj, 'growth') and obj.growth.length <= 0.0:
                obj.growth.length = 0.0001

    @staticmethod
    def get_trunk_branches(branch_hierarchy: dict) -> list[BasicWood]:
        """ Find all the buds on the trunk that have branches growing from them"""
        branches = []
        for trunks in branch_hierarchy["root"]:
            for bud_site in branch_hierarchy[trunks.name]:
                if bud_site.branch_child:
                    branches.append(bud_site.branch_child)
        return branches
    # Heavily commented since I won't be continuing my work in the lab. Hopefully this and the little write up is helpful - Marcus
    def get_energy_matrix(self, branches: list, flagged_branch_ids: list[int] = None) -> tuple[np.ndarray, list[int], list]:
        """
        Calculate the energy matrix for optimal branch-to-wire assignment.
        Evaluates strictly active BasicWood branches (untied limbs). Buds are ignored.
        """
        
        # Python safety measure: If no flagged branches are passed by the LTR script, 
        # initialize an empty list to prevent TypeErrors during the wire checks.
        if flagged_branch_ids is None:
            flagged_branch_ids = []

        # 1: Canidate Branch Pool
        open_branches = []
        
        # Loop through every object passed into the function
        for branch in branches:
            
            
            # 1. hasattr(branch, "tying"): Buds do not have tying data. This filters them out.
            # 2. not branch.tying.is_tied: Ensures we only look at completely unanchored branches.
            if hasattr(branch, "tying") and not branch.tying.is_tied:
                
                # Ensure the branch is at least half as long as the gap between wires. 
                # If it's too short to reach the trellis, don't bother evaluating it.
                if branch.growth.length > 0.5 * self.support.spacing_wires:
                    open_branches.append(branch)

        # 2: Build the avalaibale wire pool
        wire_ids = []
        
        # Loop through every physical wire object currently on the 3D trellis
        for wire_id, wire in enumerate(self.branch_attractor):
            
            # Availability Check:
            # -1 means the wire is completely empty.
            # flagged_branch_ids contains the ID of the large branch the LTR script wants to mark.
            # If the wire is held by a flagged branch, we temporarily treat the wire as "open".
            if wire.branch_id == -1 or wire.branch_id in flagged_branch_ids:
                wire_ids.append(wire_id)
            else:
                #print(f"Wire {wire_id} tied to {wire.branch_id}")
                pass

        # 3: GRID
        # Get the total counts to define the size of our 2D grid
        num_branches = len(open_branches)
        num_wires = len(wire_ids)

        # Create a NumPy grid (rows = branches, columns = wires).
        # Fill every single cell with 1000 (self.invalid_attractor_value).
        # This acts as an infinite penalty so SciPy ignores incompatible pairings.
        energy_matrix = np.full((num_branches, num_wires), self.invalid_attractor_value)

        # If the LTR script feeds us a tree with zero untied candidates, 
        # instantly return the empty matrix to prevent the SciPy solver from crashing.
        if num_branches == 0 or num_wires == 0:
            return energy_matrix, wire_ids, open_branches

        # Calculate the absolute lowest physical point on the trunk where tying is allowed.
        # This gives a small buffer zone (half a wire's spacing) below the bottom wire.
        min_start_height = self.config.start_height - self.config.spacing_wires * 0.5
        
        # 4: Evaluate the combinations
        # Loop through every valid untied branch
        for branch_idx, branch in enumerate(open_branches):
            
            # Extract the exact 3D starting coordinates (X, Y, Z) into a NumPy array
            branch_start = np.array(branch.location.start)
            # Extract the directional vector (where the branch is naturally pointing)
            branch_dir = np.array(branch.location.start_dir)
            # Find out if this trellis pulls horizontally (V-Trellis) or vertically (UFO)
            tie_type = branch.tying.tie_type

            # Height Rejection: Check the Z-axis ([2]). If the branch originates 
            # below the minimum height, skip it. Its row remains filled with 1000s.
            if branch_start[2] < min_start_height:
                continue

            # Axis Selection based on Trellis Architecture:
            if tie_type == TyingState.TyingType.TIE_ACROSS:
                # V-Trellis pulls horizontally. Target the X-axis (index 0) for distance math.
                check_coord = 0  
                spacing = self.support.spacing_across_wire()
            else:
                # Standard Trellis pulls vertically. Target the Z-axis (index 2) for distance math.
                check_coord = 2  
                spacing = self.support.spacing_wires

            # Inner Loop: Compare the current branch against every available wire
            for wire_idx, wire_id in enumerate(wire_ids):
                
                # Grab the physical wire object and its coordinates
                wire = self.branch_attractor[wire_id]
                wire_points = np.array(wire.attractor_pts)
                
                # Create a 3D arrow (vector) pointing from the branch's base exactly to the wire
                vec_to_wire_pt = wire_points[0, 0:3] - branch_start
                
                # Measure the literal physical length of that arrow in 3D space
                len_vec_to_wire = np.linalg.norm(vec_to_wire_pt)
                
                # If the branch isn't already sitting perfectly on the wire (length is not 0.0)
                if not np.isclose(len_vec_to_wire, 0.0):
                    # Divide the vector by its own length. This should normalizes the vector???
                    # down to a length of exactly 1.0, which im using to calculating angles.
                    vec_to_wire_pt /= len_vec_to_wire
                else:
                    # If it's already touching, just adopt the wire's natural direction
                    vec_to_wire_pt = wire.attractor_dir

                # Use a dot product to compare the direction the branch naturally grows 
                # against the direction it MUST be pulled to reach the wire.
                align_growth = np.dot(branch_dir, vec_to_wire_pt)
                
                # A score of < 0.45 means the branch would have to be bent sharper than ~63 degrees.
                # Reject the pairing and move to the next wire.
                if align_growth < 0.45:
                    continue

                # Compare the branch's natural growth direction against the flow of the entire trellis.
                align_wire = np.dot(branch_dir, wire.attractor_dir)
                
                # If the score is negative, the branch is growing backward relative to the row. Reject it.
                if align_wire < 0.0:
                    continue

                # Isolate the 1D distance between the branch and wire on the chosen axis (X or Z)
                start_distance_energy = branch_start[check_coord] - wire_points[0, check_coord]

                # Get the absolute value. If the physical gap is larger 
                # than 66% (2/3rds) of the distance to the next wire, it's out of bounds. Reject it.
                if np.fabs(start_distance_energy) > 2.0 * spacing / 3.0:
                    continue
                    
                # 5: Final Score
                # If the branch survives all rejections, convert the physical distance into a percentage.
                dist_energy = np.fabs(start_distance_energy) / spacing
                
                # Balance the equation: multiply the distance percentage and the angle alignment 
                # by the custom weights defined in your tree's configuration file.
                total_energy = self.config.energy_distance_weight * dist_energy + align_growth * self.config.energy_angle_weight

                # Overwrite the 1000 penalty in the grid with this final, viable float score.
                energy_matrix[branch_idx, wire_idx] = total_energy

        # Pass the fully populated grid and lists back out to decide_guide or test_assignment_viability
        return energy_matrix, wire_ids, open_branches

    def decide_guide(self, energy_matrix, wire_ids: list, branches):
        """
        Perform assignment of branches to wires based on energy matrix.

        This function uses the linear_sum_assignment method in scipy to find the optimal minimum energy assignment.
        Once a branch is assigned to a wire, both that branch and wire are marked as unavailable to prevent further assignments.

        Note that this is not a 1-1, onto assignment - it only uses reasonable pairings (see energy matrix above)
        Args:
            energy_matrix: numpy.ndarray of shape (num_branches, num_wires) with energy costs
            wire_ids: Ids of wires that can be tied
            branches: List of branch objects to be assigned

        Returns:
            None: Modifies branches and wire_attractors in-place with new assignments
        """
        # Read the dimensions of the NumPy grid. 
        # Rows become num_branches, columns become num_wires.
        num_branches, num_wires = energy_matrix.shape

        # Early return safety valve: If the matrix is empty (0 branches or 0 wires), 
        # instantly exit the function so the SciPy algorithm doesn't crash.
        if num_branches == 0 or num_wires == 0:
            return

        # This evaluates the entire grid at once to find 
        # the global optimal combination of branch-to-wire pairings with the lowest total score.
        # It outputs two parallel arrays: the row indices (branches) and column indices (wires).
        row_ind, col_ind = linear_sum_assignment(energy_matrix)
        #print(f"Energy matrix {energy_matrix}")

        # Wires are organized from left to right (ufo) or up to down (envy);
        # Continue making assignments as long as energy matrix < 1000

        # zip() binds the row and column arrays together so we can iterate through the proposed pairs.
        for branch_indx, wire_indx in zip(row_ind, col_ind):
            
            # Map the localized matrix column index (e.g., Column 1) 
            # back to the global trellis wire ID (e.g., Wire #50)
            wire_id = wire_ids[wire_indx]
            
            # Look up the original score of this specific pairing in the grid.
            # If the score is 1000 (invalid_attractor_value), it means the pairing is won't work
            # and the script skips it. It only proceeds if the score is < 1000.
            if energy_matrix[branch_indx, wire_indx] < self.invalid_attractor_value:
                #print(f" Assigning branch {branches[branch_indx].name} to wire {self.branch_attractor[wire_id].attractor_pts[0]} {energy_matrix[branch_indx, wire_indx]}")
                
                # Use the indices to grab the physical branch object and the physical wire object from memory
                branch = branches[branch_indx]
                wire_attach = self.branch_attractor[wire_id]

                # Perform the assignment 
                # Store the wire object inside the branch's tying state. When the 3D geometry generates, 
                # it will read this variable and bend the branch cylinder to reach the wire.
                branch.tying.wire_attach = wire_attach
                
                # Parse the string name of the branch (e.g., "PrimaryBranch_45") 
                # Split it at the underscore, grab the last part ("45"), and convert it to an integer (45).
                branch_id = int(branch.name.split("_")[-1])
                
                # Perform the assignment 
                # Overwrite the wire's default '-1' empty ID with the branch's ID. 
                # This locks the wire, preventing the energy matrix from evaluating it in future years.
                wire_attach.branch_id = branch_id
    def test_assignment_viability(self, energy_matrix, wire_ids: list, branches: list) -> dict:
        """
        Runs the algorithm without mutating objects to validate viability.
        Returns a dictionary mapping branch names to their proposed wire IDs.
        """
        # Read the dimensions of the NumPy grid passed in by the LTR script.
        num_branches, num_wires = energy_matrix.shape
        
        # Early return safety valve: If the grid is empty, return a blank dictionary. 
        # This prevents crashes and fulfills the function's promise to return a dict object.
        if num_branches == 0 or num_wires == 0:
            return {}

        # Run the exact SciPy to find the optimal mathematical pairings.
        row_ind, col_ind = linear_sum_assignment(energy_matrix)
        
        # Initialize a blank dictionary. 
        # to record the hypothetical ties without altering the actual tree.
        proposed_ties = {}

        # Iterate through the proposed pairs generated by the algorithm.
        for branch_indx, wire_indx in zip(row_ind, col_ind):
            
            # Ensure the pairing is  possible (score < 1000). 
            # If the score is exactly 1000, Python  skips the pairing 
            if energy_matrix[branch_indx, wire_indx] < self.invalid_attractor_value:
                
                # Grab the physical branch object using the row index
                branch = branches[branch_indx]
                # Map the local column index to the global wire ID
                wire_id = wire_ids[wire_indx]
                
                # Non-Destructive Logging: 
                # Create a new entry in the dictionary. Use the branch's string name (e.g., "PrimaryBranch_12") 
                # as the key, and set the global wire ID (e.g., 6) as the value.
                proposed_ties[branch.name] = wire_id
                
        # Hand the completed blueprint back 
        # if its specific target wire successfully secured a replacement branch.
        return proposed_ties
