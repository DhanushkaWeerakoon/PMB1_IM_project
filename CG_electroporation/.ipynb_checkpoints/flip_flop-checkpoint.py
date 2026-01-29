"""
Lipid flip-flop analysis module for membrane simulations.

Module for detecting and tracking lipid flip-flop events
(lipid translocation between membrane leaflets) over the course of a trajectory.
Monitors the number of lipids of each type in each leaflet to identify when
lipids move from one leaflet to another.
"""


import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
import pickle

def pickle_files(filename,variable):
    """
    Save a Python variable to a pickle file.
    
    Parameters
    ----------
    filename : str
        Path to output pickle file.
    variable : any
        Python object to serialize and save.
        
    Notes
    -----
    Writes file in binary mode ('wb').
    """
    
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

def Flip_flop_calc(universe,phosphates):
    """
    Count lipids of each type in both leaflets for current frame.
    
    Uses LeafletFinder to identify lipid leaflets based on phosphate positions,
    then counts the number of POPE and POPG lipids in each leaflet. Changes in
    these counts over time indicate flip-flop events.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing the membrane system.
    phosphates : AtomGroup
        Phosphate atoms used for leaflet identification in single membrane.
        
    Returns
    -------
    Leaflet0_POPE_num : int
        Number of POPE lipids in leaflet 0 (upper leaflet).
    Leaflet1_POPE_num : int
        Number of POPE lipids in leaflet 1 (lower leaflet).
    Leaflet0_POPG_num : int
        Number of POPG lipids in leaflet 0 (upper leaflet).
    Leaflet1_POPG_num : int
        Number of POPG lipids in leaflet 1 (lower leaflet).
        
    Notes
    -----
    - Uses LeafletFinder with periodic boundary conditions
    - Leaflet 0: typically the upper leaflet (groups(0))
    - Leaflet 1: typically the lower leaflet (groups(1))
    - Counts residues, not atoms, to get number of lipid molecules
    - Uses updating=True to handle dynamic leaflet assignment
    """

    # Identify leaflets using LeafletFinder
    L=LeafletFinder(universe,phosphates,pbc=True)
    Leaflet0=L.groups(0) # Upper leaflet
    Leaflet1=L.groups(1) # Lower leaflet

    # Count POPE lipids in each leaflet
    Leaflet0_POPE_num=len(Leaflet0.select_atoms("resname POPE",updating=True).residues)
    Leaflet0_POPG_num=len(Leaflet0.select_atoms("resname POPG",updating=True).residues)

    Leaflet1_POPE_num=len(Leaflet1.select_atoms("resname POPE",updating=True).residues)
    Leaflet1_POPG_num=len(Leaflet1.select_atoms("resname POPG",updating=True).residues)
        
    return Leaflet0_POPE_num,Leaflet1_POPE_num,Leaflet0_POPG_num,Leaflet1_POPG_num

class Flip_flop:
    """
    Track lipid flip-flop events over a molecular dynamics trajectory.
    
    This class monitors the distribution of lipids between membrane leaflets
    over time to detect flip-flop events (lipid translocation from one leaflet
    to another).
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    phosphate_sel : str
        MDAnalysis selection string for phosphate atoms used in leaflet
        identification. 
    midpoint : float
        Z-coordinate (Å) separating upper and lower bilayers in double bilayer 
        systems. Used to identify bilayer.
    upper : bool
        If True, analyze upper bilayer (z > midpoint).
        If False, analyze lower bilayer (z < midpoint).
    selname : str
        Base name for output pickle file.
    saving_folder : str
        Directory path where flip-flop results will be saved.
    start : int
        First frame to analyze.
    stop : int
        Last frame to analyze.
    step : int, default=1
        Analyze every nth frame.
    verbose : bool, default=True
        Print progress information during analysis.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory with continuous unwrapping enabled.
    savingfolder : str
        Directory for saving results.
    selname : str
        Base filename for outputs.
        
    
    Notes
    -----
    Workflow:
    1. Identify bilayer to analyze using midpoint parameter
    2. At each frame, use LeafletFinder to assign lipids to leaflets
    3. Count POPE and POPG lipids in each leaflet
    4. Track changes in counts over time
    5. Save results to pickle file
    
    Output Structure:
    results.results is a list of tuples, one per frame:
    [(POPE_leaf0, POPE_leaf1, POPG_leaf0, POPG_leaf1), ...]
    - Index 0: POPE in leaflet 0 (upper)
    - Index 1: POPE in leaflet 1 (lower)
    - Index 2: POPG in leaflet 0 (upper)
    - Index 3: POPG in leaflet 1 (lower)
    
    
    Limitations:
    - Only tracks POPE and POPG (can be extended to other lipid types)
    - Requires clear leaflet separation - fails with pores and highly curves/disrupted membranes.
    """
    def __init__(self,tpr,traj,phosphate_sel,midpoint,upper,selname,saving_folder,start,stop,step=1,verbose=True):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj,continuous=True)
        self.phosphate_sel=phosphate_sel
        self.midpoint=midpoint
        self.upper=upper
        self.verbose=verbose
        self.start=start
        self.stop=stop
        self.step=step
        self.savingfolder=saving_folder
        self.selname=selname
    

    def flip_flop_calc(self):
         """
        Calculate lipid distribution between leaflets over trajectory.
        
        Analyzes the specified bilayer (upper or lower based on midpoint)
        to track the number of POPE and POPG lipids in each leaflet at
        each frame. Results can be used to detect flip-flop events by
        identifying changes in lipid counts.
        
        Returns
        -------
        Flip_flops_results : AnalysisFromFunction results object
            Results object containing:
            - results : list of tuples
                List with one tuple per analyzed frame, each containing:
                (POPE_leaflet0, POPE_leaflet1, POPG_leaflet0, POPG_leaflet1)
            - times : array
                Time values (ps) for each analyzed frame
        
        Notes
        -----
        Workflow:
        1. Move to first frame of trajectory
        2. Identify bilayer to analyze using midpoint:
           - If upper=True: select phosphates with z > midpoint
           - If upper=False: select phosphates with z < midpoint
        3. Create analysis class from Flip_flop_calc function
        4. Run analysis over specified frame range
        5. At each frame, LeafletFinder assigns lipids to leaflets
        6. Count POPE and POPG in each leaflet
        7. Save results to pickle file
        
        Bilayer Selection:
        - midpoint separates upper and lower bilayers
        - Useful for systems with multiple bilayers or periodic images
        - Selected phosphates define which bilayer to analyze
        
        Leaflet Assignment:
        - LeafletFinder identifies leaflets based on phosphate clustering
        - Leaflet 0: typically upper leaflet of selected bilayer
        - Leaflet 1: typically lower leaflet of selected bilayer
        - Assignment can vary frame-to-frame for lipids near midplane
        """

        # Move to first frame
        self.universe.trajectory[0]

        # Select phosphates for bilayer of interest
        Phosphate_upper_bilayer=self.universe.select_atoms(f"({self.phosphate_sel}) and (prop z > {self.midpoint})")
        Phosphate_lower_bilayer=self.universe.select_atoms(f"({self.phosphate_sel}) and (prop z < {self.midpoint})")
        
        # Create analysis class from function
        Calculate_Flips=analysis_class(Flip_flop_calc)
        
        # Initialize and run analysis
        if self.upper==True:
            Flip_flops=Calculate_Flips(universe=self.universe,phosphates=Phosphate_upper_bilayer)
        
        if self.upper==False:
            Flip_flops=Calculate_Flips(universe=self.universe,phosphates=Phosphate_lower_bilayer)
        Flip_flops_results=Flip_flops.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)
           
        # Save results to pickle file
        pickle_files(f"{self.savingfolder}{self.selname}",[Flip_flops_results.results])
        return Flip_flops_results