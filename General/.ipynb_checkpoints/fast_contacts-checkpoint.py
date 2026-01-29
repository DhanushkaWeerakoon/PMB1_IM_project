"""
Module for calculating the total number of atomic contacts between two molecular selections over the course of a trajectory. Uses efficient KD-tree based neighbor searching with periodic boundary conditions, and includes special handling for membrane leaflet-specific lipid selections.
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
import MDAnalysis.lib.pkdtree as pkdtree
from MDAnalysis.analysis.leaflet import LeafletFinder


class Contacts(AnalysisBase):
    """
    Calculate total atomic contacts between two selections over trajectory.
    
    This class computes the number of contacts between atoms in two selections
    at each frame. For each atom in selection B, it counts how many atoms in
    selection A are within the proximity cutoff. The total contact count is
    summed across all atoms in selection B. Includes automatic leaflet 
    identification for lipid selections.

    The code was adapted from the work of Jonathan Shearer: 
        https://github.com/js1710/LPS_fingerprints_paper/blob/master/membrane_analysis/
        lipid_lifetime.py

    This class inherits from MDAnalysis.AnalysisBase class.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    selA : str
        MDAnalysis selection string for first group (reference group).
        If this is a lipid selection (POPE, POPG, or RAMP), only the
        upper leaflet will be selected.
    selB : str
        MDAnalysis selection string for second group (query group).
    proximity : float, default=6.0
        Distance cutoff (Å) for defining a contact.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    Selection_A : AtomGroup
        Selected atoms from selA (automatically filtered to upper leaflet
        if selA is a lipid selection).
    Selection_B : AtomGroup
        Selected atoms from selB.
    Contacts_over_time : ndarray
        Array of total contact counts at each analyzed frame.
        Shape: (n_frames,).
    phosphates : str
        Selection string for phosphate atoms used in leaflet identification.
        Default: "(resname RAMP and name PA PB) or (resname POPE POPG and (name P))"
        
    
    Notes
    -----
    - Uses periodic KD-tree for efficient neighbor searching with PBC
    - Total contacts = sum of contacts across all atoms in selB
    - Automatic leaflet filtering: if selA is "resname POPE", "resname POPG",
      or "resname RAMP", only upper leaflet atoms are selected
    - KD-tree is rebuilt at each frame for accurate distance calculations
    
    Differences from Intermolecular_contacts class
    -----------------------------------------------
    - This class counts contacts per atom in selB, then sums them
    - Intermolecular_contacts counts unique atom pairs within cutoff
    - This class uses KD-tree for potentially faster searching
    - This class has automatic leaflet filtering for lipids
    - Results may differ due to different counting methodologies
    

    """
    
    def __init__(self,tpr,traj,selA,selB,proximity=6.0):
        universe=mda.Universe(tpr,traj,continuous=True)
        super(Contacts,self).__init__(universe.trajectory)
        self.tpr=tpr
        self.traj=traj
        self.universe=universe
        self.selA=selA
        self.selB=selB
        self.proximity=proximity

        # Define phosphate selection for leaflet identification
        self.phosphates="(resname RAMP and name PA PB) or (resname POPE POPG and (name P))" 

        # Handle lipid selections with automatic leaflet filtering
        if self.selA=="resname POPE" or self.selA=="resname POPG" or self.selA=="resname RAMP":
            L=LeafletFinder(self.universe,self.phosphates,pbc=True)
            leaflet0=L.groups(0)
            leaflet1=L.groups(1)
            # Select only upper leaflet atoms
            self.Selection_A=leaflet0.residues.atoms.select_atoms(self.selA)
        else:
            # Non-lipid selection: use entire universe
            self.Selection_A=self.universe.select_atoms(f"{self.selA}")
        self.Selection_B=self.universe.select_atoms(f"{self.selB}")
   
    def _prepare(self):
        """
        Initialize contact storage list before analysis.
        
        Called automatically by AnalysisBase.run() before iteration begins.
        Creates empty list to store total contact counts for each frame.
        """
        self.Contacts_over_time=[]

    def search(self,selA,selB,cutoff,level="A"):
        """
        Search for atoms in selA within cutoff distance of selB.
        
        Uses a periodic KD-tree for efficient neighbor searching with
        periodic boundary conditions.
        
        Parameters
        ----------
        selA : AtomGroup
            Reference atom group to search within.
        selB : Atom
            Single query atom to search around.
        cutoff : float
            Distance cutoff (Å) for neighbor search.
        level : str, default="A"
            Level of grouping for results:
            - "A": return atoms
            - "R": return residues
            - "S": return segments
            
        Returns
        -------
        list
            List of atoms, residues, or segments within cutoff distance,
            depending on level parameter.
            
        Notes
        -----
        Uses self.kdtree which must be initialized before calling.
        """

        positions=selB.position
        unique_idx=self.kdtree.search(positions,cutoff)
        return self._index2level(selection=selA,indices=unique_idx,level=level)

    def _index2level(self,selection,indices,level):
        """
        Convert atom indices to atoms, residues, or segments.
        
        Parameters
        ----------
        selection : AtomGroup
            Atom group containing the atoms of interest.
        indices : array-like
            Indices of atoms within selection.
        level : str
            Grouping level: "A" (atoms), "R" (residues), or "S" (segments).
            
        Returns
        -------
        list
            List of atoms, unique residues, or unique segments depending
            on level parameter.
            
        Raises
        ------
        NotImplementedError
            If level is not one of "A", "R", or "S".
        """
        
        n_atom_list=selection[indices]
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
            raise NotImplementedError('{0}: level not implemented.'.format(level))

    def _single_frame(self):
        """
        Calculate total contacts for current frame.
        
        For each atom in Selection_B, counts how many atoms in Selection_A
        are within the proximity cutoff. Sums these counts across all atoms
        in Selection_B to get the total contact count for the frame.
        
        Called automatically by AnalysisBase.run() for each frame.
        
        Workflow
        --------
        1. Build KD-tree for Selection_A positions at current frame
        2. For each atom in Selection_B:
           a) Search for Selection_A atoms within proximity cutoff
           b) Count number of contacts found
        3. Sum contact counts across all Selection_B atoms
        4. Append total to Contacts_over_time list
        
        Notes
        -----
        - KD-tree includes 0.1 Å buffer for initialization
        - Periodic boundary conditions handled by PeriodicKDTree
        
       
        """
        
        List=[]

        # Build KD-tree for current frame
        self.kdtree=pkdtree.PeriodicKDTree(box=self.universe.dimensions)
        self.kdtree.set_coords(self.Selection_A.positions,cutoff=self.proximity+0.1)

        # Count contacts for each atom in Selection_B
        for atom in self.Selection_B:
            contacts=self.search(selA=self.Selection_A,selB=atom,cutoff=self.proximity,level="A")
            List.append(len(contacts))

        # Store total contact count for this frame
        self.Contacts_over_time.append(np.sum(List))
        
    
    def _conclude(self):
        """
        Finalize analysis after all frames processed.
        
        Converts Contacts_over_time list to numpy array for efficient
        numerical operations and analysis.
        
        Called automatically by AnalysisBase.run() after processing all frames.
        
        Sets self.Contacts_over_time to numpy array of shape (n_frames,).
        """
        
        self.Contacts_over_time=np.array(self.Contacts_over_time)

