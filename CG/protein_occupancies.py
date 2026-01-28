"""
Module for calculating residue-level occupancies i.e. whether protein residues are within contact with another chemical moiety (e.g. lipids) in a given simulation frame.
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
import MDAnalysis.lib.pkdtree as pkdtree


class Occupancy(AnalysisBase):
    """
    This class determines which residues in a selection (typically protein)
    are within a specified distance of another selection (e.g., lipids)
    at each frame. Results indicate whether each residue has any contacts
    (binary: 1 = contact, 0 = no contact).

    This class inherits from the MDAnalysis AnalysisBase class.

    
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    selA : str
        MDAnalysis selection string for the reference group (e.g., lipids).
        This is the group that contacts are measured to.
    selB : str
        MDAnalysis selection string for the residue group (e.g., protein).
        This is typically the protein whose residues are being tracked.
    offset : int, default=0
        Residue ID offset. Use this if residue numbering doesn't start at 0.
        For example, if residues start at 100, set offset=99.
    protein_residues : int, default=417
        Number of residues in selB to analyze.
    proximity : float, default=6.0
        Distance cutoff (Å) for defining a contact.
    start : int, default=0
        First frame to analyze.
    stop : int, default=-1
        Last frame to analyze (-1 for all frames).
    step : int, default=1
        Analyze every nth frame.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    Selection_A : AtomGroup
        Selected atoms from selA (reference group).
    Selection_B : AtomGroup
        Selected atoms from selB (residue group).
    Residue_list : list of AtomGroups
        List containing an AtomGroup for each residue in the range.
    Contacts_over_time : ndarray
        Array of shape (n_frames, protein_residues) where elements can take values of 1
        if residue has a contact and 0 otherwise.
        

    Notes
    -----
    - Uses periodic KD-tree for efficient neighbor searching.
    - Contact is defined as any atom-atom distance < proximity cutoff
    - Residue IDs are adjusted by offset parameter (actual_resid = index + 1 + offset)
    """
    def __init__(self,tpr,traj,selA,selB,offset=0,protein_residues=417,proximity=6.0,start=0,stop=-1,step=1):
        universe=mda.Universe(tpr,traj,continuous=True)
        super(Occupancy,self).__init__(universe.trajectory)
        self.tpr=tpr
        self.traj=traj
        self.universe=universe
        self.selA=selA
        self.selB=selB
        self.protein_residues=protein_residues
        self.proximity=proximity
        self.offset=offset

        # Select atom groups
        self.Selection_A=self.universe.select_atoms(f"{self.selA}")
        self.Selection_B=self.universe.select_atoms(f"{self.selB}")

        # Create list of individual residue selections
        self.Residue_list=[self.universe.select_atoms(f"{self.selB} and resid {i+1+self.offset}") for i in range(self.protein_residues)]

    def _prepare(self):
        """
        Initialize analysis arrays before running.
        
        Creates empty list to store contact data for each frame.
        Called automatically by AnalysisBase.run().
        """
        self.Contacts_over_time=[]

    def search(self,selA,selB,cutoff,level="A"):
        """
        Search for atoms in selA within cutoff distance of selB.
        
        Uses a periodic KD-tree for efficient neighbor searching with
        periodic boundary conditions.

        The function was adapted from the work of Jonathan Shearer: 
        https://github.com/js1710/LPS_fingerprints_paper/blob/master/membrane_analysis/
        lipid_lifetime.py
        
        Parameters
        ----------
        selA : AtomGroup
            Reference atom group to search within.
        selB : AtomGroup or Atom
            Query atom(s) to search around.
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
        positions=selB.atoms.position

        unique_idx=self.kdtree.search(positions,cutoff)
        return self._index2level(selection=selA,indices=unique_idx,level=level)

    def _index2level(self,selection,indices,level):
        """
        Groups atoms in selection by level (atoms, residues or segments).

        The function was adapted from the work of Jonathan Shearer: 
        https://github.com/js1710/LPS_fingerprints_paper/blob/master/membrane_analysis/
        lipid_lifetime.py
        
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
        For each residue in the protein, determines if any of its atoms
        are within proximity distance of any atoms in Selection_A.
        Updates Contacts_over_time with binary contact statuses of each residue.
        
        Called automatically by AnalysisBase.run() for each frame.
        
        Notes
        -----
        - Creates a KD-tree for Selection_A (non-protein) positions at current frame
        - Iterates through all residues and their atoms
        - Marks residue as 1 (contact) as soon as any atom contact is found
        - Contact cutoff includes 0.1 Å buffer for KD-tree initialization
        """
        # Initialise contact histogram
        resname_hist={}

        # Build kdtree 
        for i in range(self.protein_residues):
            resname_hist[i+1+self.offset]=0
        
        self.kdtree=pkdtree.PeriodicKDTree(box=self.universe.dimensions)
        self.kdtree.set_coords(self.Selection_A.positions,cutoff=self.proximity+0.1)

        # Check for contacts between protein residues and non-protein moieties at the atom level.
        for (i,j) in zip(self.Residue_list,range(self.protein_residues)):
            for atom in i:
                near=self.search(selA=self.Selection_A,selB=atom,cutoff=self.proximity, level="A")
                if len(near) ==0:
                    continue
                else:
                    resname_hist[j+1+self.offset]=1
                    break

        # Store binary contact status for all residues.
        residues=[j for i,j in resname_hist.items()]
        self.Contacts_over_time.append(residues)
   
       
        
    
    def _conclude(self):
        """        
        Converts self.Contacts_over_time list to numpy array of shape 
        (n_frames, protein_residues).
        Called automatically by AnalysisBase.run() after processing all frames.
        """
        self.Contacts_over_time=np.array(self.Contacts_over_time,dtype='object')


