"""
Module for calculating residence times (lifetimes) of lipids near proteins or peptides and computing survival probabilities. Uses efficient KD-tree based neighbour searching with periodic boundary conditions and Numba-accelerated probability calculations for analyzing long-timescale binding events.

The module and associated functions were adapted from the work of Jonathan Shearer: 
https://github.com/js1710/LPS_fingerprints_paper/blob/master/membrane_analysis/lipid_lifetime.py

Module dependencies:
- tqdm: Progress meter for loops
  Example: for i in tqdm(range(10000)): ...

- KDTree: Scipy package for quick nearest-neighbour lookup
  Provides index into k-dimensional points for rapid neighbor searches
  MDAnalysis provides wrappers for PBC-aware searches

- Numba: JIT compiler translating Python to optimized machine code
  Uses LLVM compiler for C/FORTRAN-like performance
  Applied via @jit decorator for numerical algorithms
"""

import numpy as np
import tqdm
import MDAnalysis as mda
import MDAnalysis.lib.pkdtree as pkdtree
from MDAnalysis.core.groups import Atom, AtomGroup
import argparse
from numba import jit


def distance_sq(pos1, pos2):
    """
    Calculate squared Euclidean distance between two positions.
    
    Parameters
    ----------
    pos1 : ndarray
        First position vector, shape (3,).
    pos2 : ndarray
        Second position vector, shape (3,).
        
    Returns
    -------
    float
        Squared distance between positions.
        
    Notes
    -----
    Squared distance avoids expensive sqrt operation.
    For distance comparisons, squared distances are sufficient.
    """
    diff = pos2 - pos1
    return np.sum(np.square(diff))

@jit(nopython=True)
def survival_prob(lipids, dt):
    """
    Calculate survival probability for lipid-protein binding.
    
    Computes the probability that a lipid remains bound to a protein
    after a time lag dt, given that it was bound at an initial time.
    Uses Numba JIT compilation for fast computation.
    
    Parameters
    ----------
    lipids : ndarray
        2D array where each row represents a lipid and each column
        represents a frame. Value is 1 if lipid is bound, 0 otherwise.
        Shape: (n_lipids, n_frames).
    dt : int
        Time lag in frames. Number of frames to look ahead when
        checking if binding is maintained.
        
    Returns
    -------
    bound : int
        Number of instances where lipid remains bound after time dt.
    total : int
        Total number of instances where lipid was initially bound.
        
    Notes
    -----
    Survival probability P(dt) = bound / total
    
    For each lipid and each frame where it's bound (lipid[i] == 1):
    - Check if still bound dt frames later (lipid[i + dt] == 1)
    - If yes: increment bound counter
    - Always increment total counter
    
    The survival probability decays exponentially for simple binding:
    P(t) = exp(-t/τ) where τ is the residence time
    
    Uses @jit(nopython=True) for ~100x speedup on large datasets.
    
    """
    bound = 0
    total = 0
    for i in range(len(lipids)):
        lipid = lipids[i]
        for i in range(len(lipid) - dt):
            if lipid[i] == 1:
                if lipid[i + dt] == 1:
                    bound += 1
                total += 1

    return bound, total

class NeighbourSearchFast(object):
    """
    Fast periodic neighbor searching using KD-tree.
    
    Wrapper class for efficient neighbor searches with periodic boundary
    conditions using MDAnalysis's PeriodicKDTree. Provides rapid lookup
    of atoms/residues/segments within a specified radius.
    
    Parameters
    ----------
    atomgroup : AtomGroup
        MDAnalysis AtomGroup to search within (reference group).
    box : array-like, optional
        Simulation box dimensions. If None, uses universe dimensions.
        
    Attributes
    ----------
    atomgroup : AtomGroup
        Reference atom group for searches.
    box : array-like
        Box dimensions for periodic boundary conditions.
    kdtree : PeriodicKDTree
        KD-tree object for efficient neighbor searching.
    indices : ndarray
        Indices of atoms found in most recent search.
    
    Notes
    -----
    - Uses PeriodicKDTree for PBC-aware distance calculations
    - Tree must be rebuilt (set_coords) for each new frame
    - Default leafsize=10 balances tree construction vs search time
    - Coordinates automatically wrapped to primary unit cell
    """
            
        
    def __init__(self, atomgroup, box=None):
        """
        Initialize neighbor search object.
        
        Creates KD-tree structure for efficient neighbor searching
        with periodic boundary conditions.
        
        Notes
        -----
        Tree coordinates are not set at initialization - must call
        search() which internally calls set_coords().
        """
        
        self.atomgroup = atomgroup
        self._u = atomgroup.universe
        self.box = box
        self.kdtree = pkdtree.PeriodicKDTree(box=self.box)


    def search(self, searchgroup, radius,level="A"):
        """
        Search for neighbors within radius of searchgroup.
        
        Builds KD-tree from atomgroup positions and searches for all
        atoms within the specified radius of searchgroup atoms (or atom).
        
        Parameters
        ----------
        searchgroup : AtomGroup or Atom
            Query atoms or single atom to search around.
        radius : float
            Search radius in Angstroms.
        level : str, default="A"
            Grouping level for results:
            - "A": return atoms
            - "R": return residues (unique)
            - "S": return segments (unique)
            
        Returns
        -------
        list
            List of atoms, residues, or segments within radius,
            depending on level parameter.
            
        Notes
        -----
        - Rebuilds KD-tree for current atomgroup positions
        - Uses radius + 0.1 Å buffer for tree construction
        - Handles both single Atom and AtomGroup searches
        - Coordinates wrapped to primary cell for PBC
        - Results include periodic images
        - Sets self.indices with atom indices found
        
        """

        # Build KD-tree from current atomgroup positions
        self.kdtree.set_coords(self.atomgroup.positions,cutoff=radius + 0.1)

        # Handle single Atom vs AtomGroup
        if isinstance(searchgroup, Atom):
            positions = searchgroup.positions.reshape(1, 3)
        else:
            positions = searchgroup.positions

        # Search for atoms within radius (including periodic images)
        unique_idx = self.kdtree.search(positions, radius)
       
        # Store indices for later access
        self.indices = unique_idx
        
        # Convert indices to requested level (atoms/residues/segments)
        return self._index2level(unique_idx, level)

    def _index2level(self, indices, level):
        """
        Convert atom indices to atoms, residues, or segments.
        
        Parameters
        ----------
        indices : array-like
            Indices of atoms in atomgroup.
        level : str
            Grouping level: "A" (atoms), "R" (residues), or "S" (segments).
            
        Returns
        -------
        list
            List of atoms, unique residues, or unique segments.
            
        Raises
        ------
        NotImplementedError
            If level is not "A", "R", or "S".
        """
        n_atom_list = self.atomgroup[indices]
        if level == 'A':
            if not n_atom_list:
                return []
            else:
                return n_atom_list
        elif level == 'R':
            return list({a.residue for a in n_atom_list})
        elif level == 'S':
            return list(set([a.segment for a in n_atom_list]))
        else:
            raise NotImplementedError('{0}: level not implemented'.format(level))
    

class Lipid_survival():
    """
    Analyze lipid residence times and survival probabilities near proteins.
    
    This class calculates binding kinetics for lipid-protein interactions by
    tracking which lipids are within a cutoff distance of a protein over time.
    Computes individual binding event lifetimes, survival probability curves,
    and statistical summaries with bootstrap error estimates.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    PMB1_id : str
        Residue ID selection for peptide/protein. Format: "1:100" or "50".
        Used to identify specific peptide or protein instance.
    saving_folder : str
        Directory path where results will be saved.
    save_prefix : str, default="lifetime"
        Prefix for output filenames (currently not used in output names).
    start : int, default=0
        First frame to analyze.
    stop : int, default=-1
        Last frame to analyze (-1 for all frames).
    step : int, default=1
        Analyze every nth frame.
    lipid_select : str, default="RAMP"
        Residue name for lipids to analyze (e.g., "RAMP", "POPC", "PMB1").
    protein_select : str, default="PMB1"
        Selection string for protein/peptide (residue name or full selection).
    proximity : float, default=6.0
        Distance cutoff (Å) for defining lipid-protein contact.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    time_on : dict
        Dictionary mapping residue IDs to lists of binding lifetimes (ns).
        Format: {resid: [lifetime1, lifetime2, ...]}
    
    Notes
    -----
    Binding Event Definition:
    - Lipid is "bound" if any of its atoms are within proximity of protein
    - Binding event starts when lipid enters proximity
    - Binding event ends when lipid leaves proximity
    - Multiple binding events per lipid are tracked separately
    
    Workflow:
    1. For each frame, identify lipids within proximity cutoff
    2. Track when each lipid enters/exits binding region
    3. Record lifetime of each individual binding event
    4. Calculate survival probability P(t) = fraction still bound at time t
    5. Compute statistics with bootstrap error estimates
    
    Output Files (created by Write_files):
    1. lifetime_{lipid}_{protein}_{id}_survival.dat:
       - Survival probability vs time lag
       - Columns: [time_lag(frames), probability]
       
    2. lifetime_{lipid}_{protein}_{id}_summary.dat:
       - Statistical summary per lipid type
       - Mean lifetime ± standard error
       - Bootstrap mean ± bootstrap error
       - Maximum observed lifetime
    
    Computational Details:
    - Uses KD-tree for O(log N) neighbor searches
    - Numba JIT compilation for fast survival probability
    - Bootstrap resampling (200 iterations) for error estimates
    - Progress bars via tqdm for long calculations
    
    Physical Interpretation:
    - Exponential decay: single binding mode
    - Multi-exponential decay: multiple binding modes/sites
   
    """
    def __init__(self,tpr,traj,PMB1_id,saving_folder,save_prefix="lifetime",start=0,stop=-1,step=1,lipid_select="RAMP",protein_select="PMB1",proximity=6.0):
        self.tpr=tpr
        self.traj=traj
        self.prefix=save_prefix
        self.savingfolder=saving_folder
        self.start=start
        self.stop=stop
        self.step=step
        self.lipid_select=lipid_select
        self.protein_select=protein_select
        self.PMB1_id=PMB1_id
        self.universe=mda.Universe(tpr,traj,continuous=True)
        self.proximity=proximity
    
    def Survival_calculation(self):
        """
        Calculate lipid residence times and survival probabilities.
        
        Tracks lipid-protein binding events over the trajectory to compute
        individual binding lifetimes and frame-by-frame contact matrices.
        These are used to calculate survival probability curves.
        
        Returns
        -------
        survival_rates : dict
            Dictionary mapping residue names to contact matrices.
            Format: {resname: ndarray of shape (n_lipids, n_frames)}
            Each element is 1 if that lipid is bound at that frame, else 0.
        lifetimes : dict
            Dictionary mapping residue names to lists of lifetimes.
            Format: {resname: [lifetime1, lifetime2, ...]} in nanoseconds.
            Each lifetime represents one binding event.
            
        Notes
        -----
        Algorithm:
        1. Initialize contact tracking for all lipid residues
        2. For each frame:
           a) Build KD-tree for lipids
           b) Search for lipids within proximity of protein
           c) Update contact matrix
           d) Track start/end of binding events
           e) Record lifetime when binding event ends
        3. Finalize lifetimes for still-bound lipids at trajectory end
        4. Aggregate by residue name
        
        Binding Event Tracking:
        - lipids_on_prev: list of bound lipid resids from previous frame
        - time_on: current binding duration for each bound lipid
        - lifetimes_per_resid: all completed binding events per resid
        - lipid_frame_contacts: binary matrix of contacts
        
        Timestep Calculation:
        - dt = (time[frame1] - time[frame0]) / 1000
        - Converts ps to ns
        - Assumes constant timestep
        
        Contact Matrix:
        - Shape: (n_lipids, n_frames)
        - Element [i,j] = 1 if lipid i is bound at frame j
        - Used for survival probability calculation
        
        Lifetime Calculation:
        - Each binding event tracked independently
        - Start: lipid enters proximity region
        - End: lipid exits proximity region
        - Duration accumulated in time_on dict
        - Stored when binding event ends
        
        Selection Logic:
        If lipid_select == "RAMP":
        - Select all RAMP lipids as potential binders
        - Select specific protein/peptide by resid
        
        If lipid_select == "PMB1":
        - Select specific PMB1 molecules as potential binders
        - Analyze binding to different selection (e.g., protein)
        
        """

        # Select lipids and protein based on configuration
        if self.lipid_select == "RAMP":
            lipids=self.universe.select_atoms(f"resname {self.lipid_select}")
            print(self.PMB1_id)
            protein=self.universe.select_atoms(f"resname {self.protein_select} and resid {self.PMB1_id}")
        
        elif self.lipid_select == "PMB1":
            lipids=self.universe.select_atoms(f"resname {self.lipid_select} and resid {self.PMB1_id}")
            protein=self.universe.select_atoms(f"{self.protein_select}")

        # Initialize tracking variables
        lipids_on_prev=[]
        time_on={}

        # Calculate timestep in nanoseconds
        dt=(self.universe.trajectory[1].time-self.universe.trajectory[0].time)/1000
        self.universe.trajectory[0]
        
        lifetimes_per_resid={}

        # Create contact matrix: 1 if bound, 0 if not
        # Shape: (n_lipid_residues, n_frames)
        lipid_frame_contacts=dict([(resid,np.zeros(len(self.universe.trajectory[self.start:self.stop:self.step]))) for resid in lipids.residues.resids])

        # Iterate over trajectory with progress bar
        for i, frame in enumerate(tqdm.tqdm(self.universe.trajectory[self.start:self.stop:self.step])):
            # Initialize frame contacts
            frame_contacts=dict([(resid,0) for resid in lipids.residues.resids])

            # Build KD-tree and find nearby lipids
            tree=NeighbourSearchFast(lipids,box=self.universe.dimensions)
            lipids_near=tree.search(protein,self.proximity,level='R')
            resids_on=[lipid.resid for lipid in lipids_near]
    
            # First frame: initialize tracking
            if not lipids_on_prev:
                lipids_on_prev = resids_on
                for resid in resids_on:
                    time_on[resid] = dt
                continue

            # Update binding times and record contacts
            for resid in resids_on:
                # Mark contact in matrix
                lipid_frame_contacts[resid][i] += 1
                 # If lipid was already bound, accumulate time
                if resid in lipids_on_prev:
                    time_on[resid] += dt
                # If newly bound, start new binding event
                else:
                    time_on[resid] = dt

            # Check for binding events that ended
            for resid in lipids_on_prev:
                if resid not in resids_on:
                    # Binding event ended - record lifetime
                    try:
                        lifetimes_per_resid[resid].append(time_on[resid])

                    except KeyError:
                        lifetimes_per_resid[resid] = [time_on[resid]]
                    finally:
                        del time_on[resid]
            # Update for next iteration
            lipids_on_prev = resids_on

        # Store lifetimes in class attribute
        self.time_on=lifetimes_per_resid   
        print(time_on)

        # Finalize lifetimes for lipids still bound at end
        for resid in time_on:
            try:
                lifetimes_per_resid[resid].append(time_on[resid])
            except KeyError:
                lifetimes_per_resid[resid] = [time_on[resid]]

        # Aggregate by residue name
        lifetimes = {}
        survival_rates = {}
        for resid in lifetimes_per_resid:
            resname = self.universe.select_atoms("resid " + str(resid)).residues[0].resname
            try:
                survival_rates[resname].append(lipid_frame_contacts[resid])
                lifetimes[resname].extend(lifetimes_per_resid[resid])
            except KeyError:
                lifetimes[resname] = lifetimes_per_resid[resid]
                survival_rates[resname] = [lipid_frame_contacts[resid]]
                
        # Convert to numpy arrays
        for resname in survival_rates:
            survival_rates[resname] = np.array(survival_rates[resname])
        
        return survival_rates, lifetimes
    
    def Write_files(self,survival_rates,lifetimes):
        """
        Write survival probabilities and lifetime statistics to files.
        
        Creates two output files:
        1. Survival probability vs time lag curve
        2. Statistical summary with bootstrap error estimates
        
        Parameters
        ----------
        survival_rates : dict
            Dictionary from Survival_calculation() containing contact matrices.
            Format: {resname: ndarray of shape (n_lipids, n_frames)}
        lifetimes : dict
            Dictionary from Survival_calculation() containing lifetimes.
            Format: {resname: [lifetime1, lifetime2, ...]} in nanoseconds.
            
     
        Notes
        -----
        Survival Probability Calculation:
        - For each time lag dt (0 to 49999 frames):
          - Count instances where lipid bound at frame i AND frame i+dt
          - Divide by total instances where lipid bound at frame i
          - P(dt) = probability of remaining bound after dt frames
        
        Bootstrap Error Estimation:
        - Randomly resample lifetimes 200 times
        - Each resample has same size as original dataset
        - Resampling uses weighted random selection
        - Bootstrap error = std dev of 200 resample means
        - More robust than standard error for non-normal distributions
        
        File Naming:
        - Replaces ":" with "_" in PMB1_id for valid filenames
        - Example: PMB1_id="1:10" → "1_10" in filename

        Creates files in self.savingfolder:
        
        1. lifetime_{lipid}_{protein}_{id}_survival.dat:
           Survival probability P(dt) for time lags 0-49999 frames
           Format: two columns (time_lag, probability)
           
        2. lifetime_{lipid}_{protein}_{id}_summary.dat:
           Statistical summary including:
           - Mean lifetime ± standard error
           - Bootstrap mean ± bootstrap standard deviation
           - Maximum observed lifetime
        """

        # Clean PMB1_id for filename (replace : with _)
        PMB1_id_change=self.PMB1_id.replace(":","_")

        # Write survival probability curve
        for resname in survival_rates:

            
            with open(f"{self.savingfolder}lifetime_{self.lipid_select}_{self.protein_select}_{PMB1_id_change}_survival.dat", "w") as out:
                template = "{0:^6d}{1:^8.3e}"
                
                # Calculate survival probability for each time lag
                for dt in tqdm.tqdm(range(0, 50000, 1)):
                    bound, total = survival_prob(survival_rates[resname], dt)
                try:
                    prob = bound / total
                except ZeroDivisionError:
                    prob = 0.
                    print(template.format(dt, prob), file=out)

        # Write statistical summary
        with open(f"{self.savingfolder}lifetime_{self.lipid_select}_{self.protein_select}_{PMB1_id_change}_summary.dat", "w") as out:
            for resname, times in lifetimes.items():
                # Basic statistics
                average = np.mean(times)
                maxi= np.max(times)
                error = np.std(times)
                std_error = error / np.sqrt(len(times))

                # Bootstrap error estimation
                bootstraps = []
                for n in range(200):
                    # Weighted random resampling
                    weights = np.random.random(len(times))
                    weights = weights / np.sum(weights)
                    resample = np.random.choice(times, len(times), p=weights)
                    resample_av = np.mean(resample)
                    bootstraps.append(resample_av)

                bootstraps = np.array(bootstraps)

                # Print to console
                print(len(bootstraps), len(lifetimes))
                print("Lipid {0}: {1:8.3f} +\- {2:8.3f} ns".format(resname, average, std_error))
                print("Bootstrap: {0:8.3f} +\- {1:8.3f} ns".format(np.mean(bootstraps),np.std(bootstraps)))
                print("Lipid max time = {0} ns".format(maxi))

                # Write to file
                print("Lipid {0}: {1:8.3f} +\- {2:8.3f} ns".format(resname, average, std_error), file=out)
                print("Bootstrap: {0:8.3f} +\- {1:8.3f} ns".format(np.mean(bootstraps),np.std(bootstraps) , file=out))
                print("Lipid max time = {0} ns".format(maxi),file=out)

