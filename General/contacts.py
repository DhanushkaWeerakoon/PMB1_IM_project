"""
Module for calculating the number of intermolecular contacts between two molecular selections over the course of a trajectory. Contacts are defined as atom pairs within a specified distance cutoff.

Code from MDAnalysis tutorials.
"""


import MDAnalysis as mda
from MDAnalysis.analysis import contacts, distances
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.base import AnalysisFromFunction
import numpy as np
import pickle 
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)

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

def contacts_within_cutoff(u, group_a, group_b, radius=6.0):
    """
    Count contacts between two atom groups within a distance cutoff.
    
    Calculates pairwise distances between all atoms in group_a and group_b,
    then counts how many atom pairs are within the specified radius. Takes
    periodic boundary conditions into account.
    
    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe containing the atom groups. Used to get box dimensions
        for periodic boundary correction.
    group_a : AtomGroup
        First selection of atoms.
    group_b : AtomGroup
        Second selection of atoms.
    radius : float, default=6.0
        Distance cutoff (Å) for defining a contact.
        
    Returns
    -------
    n_contacts : int
        Number of atom pairs within the distance cutoff.

    """

    
    # Calculate pairwise distances with periodic boundary conditions
    dist = distances.distance_array(group_a.positions, group_b.positions,box=u.dimensions)
    
    # Convert distance matrix into a binary contact matrix and count contacts
    n_contacts = contacts.contact_matrix(dist, radius).sum()
    return n_contacts

# Create analysis class from function for trajectory iteration
Calculate_contacts = analysis_class(contacts_within_cutoff)

class Intermolecular_contacts():
    """
    Calculate intermolecular contacts between two selections over trajectory.
    
    This class computes the number of atomic contacts between two molecular
    selections (e.g., protein and lipids, peptide and membrane) at each frame
    of a trajectory. Contacts are defined by a distance cutoff, and results
    provide time series of contact counts.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    selA : str
        MDAnalysis selection string for first group (e.g., "protein").
    selB : str
        MDAnalysis selection string for second group (e.g., "resname POPC").
    phosphates : str, default='name PO1 PO2 PO4'
        MDAnalysis selection string for phosphate atoms. Currently unused
        in the implementation but reserved for leaflet-based analysis.
    proximity : float, default=6.0
        Distance cutoff (Å) for defining a contact.
    start : int, default=0
        First frame to analyze.
    stop : int, default=-1
        Last frame to analyze (-1 for all frames).
    step : int, default=1
        Analyze every nth frame.
    verbose : bool, default=True
        Print progress information during analysis.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    results : ndarray
        Array of contact counts at each analyzed frame.
        Shape: (n_frames,).
    times : ndarray
        Array of time values (ps) corresponding to each frame.
        Shape: (n_frames,).
    
    Notes
    -----
    - Contacts are counted at the atom level (not residue level)
    - Each atom pair within cutoff counts as one contact
    - The phosphates parameter is currently unused but may be utilized in
      future versions for leaflet-specific contact analysis (see commented
      code in Contact_calculator method)
    - Results can be saved using the pickle_files function
    
    See Also
    --------
    MDAnalysis.analysis.contacts : Contact analysis tools
    MDAnalysis.analysis.distances : Distance calculation functions
    """
    
    def __init__(self,tpr,traj,selA,selB,phosphates='name PO1 PO2 PO4',proximity=6.0,start=0,stop=-1,step=1,verbose=True):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj,continuous=True)
        self.selA=selA
        self.selB=selB
        self.phosphates=phosphates
        self.proximity=proximity
        self.start=start
        self.stop=stop
        self.step=step
        self.verbose=verbose
        
    def Contact_calculator(self):
        """
        Calculate intermolecular contacts over the trajectory.
        
        Selects atoms based on selA and selB, then computes the number
        of contacts (atom pairs within proximity cutoff) at each frame.
        Results are stored in self.results and self.times attributes.
        
        Sets Attributes
        ---------------
        results : ndarray
            Number of contacts at each analyzed frame. Shape: (n_frames,).
        times : ndarray
            Time values (ps) at each analyzed frame. Shape: (n_frames,).
                    
        Workflow
        --------
        1. Select atoms based on selA (group A)
        2. Select atoms based on selB (group B)
        3. Create Calculate_contacts analysis object
        4. Run analysis over specified frame range
        5. Extract results and times from analysis object
        6. Store in self.results and self.times
        
        Alternative Implementation
        --------------------------
        Commented code shows how to restrict selB to upper leaflet:
        - Uses LeafletFinder to identify leaflets
        - Filters selB to only include upper leaflet atoms
        - Useful for analyzing contacts with specific membrane leaflet
        
        To enable leaflet-specific analysis, uncomment the relevant lines
        and adjust the conditional logic for selB selection.
        """
        
        # Optional: Identify leaflets (currently commented out)
        #L=LeafletFinder(self.universe,self.phosphates,pbc=True)
        #leaflet0=L.groups(0)
        #leaflet1=L.groups(1)

        # Select first group of atoms
        A=self.universe.select_atoms(self.selA)

        # Optional: Filter selB by leaflet (currently commented out)
        #if self.selB=="resname POPE" or self.selB=="resname POPG":
        #print(self.selB)
        #B=leaflet0.residues.atoms.select_atoms(self.selB)
        #else:
            #B=self.universe.select_atoms(self.selB)

        B=self.universe.select_atoms(self.selB)

        # Create and run contact analysis
        Contacts=Calculate_contacts(u=self.universe,group_a=A,group_b=B,radius=self.proximity)
        Contacts.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)

        # Extract and store results
        Results=Contacts.results["timeseries"]
        Times=Contacts.results["times"]
        self.results=Results
        self.times=Times
        
        
        
        
        
   
        
        

        