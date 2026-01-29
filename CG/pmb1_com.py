"""
Center of Mass (COM) analysis module for molecular dynamics simulations.

This module provides tools for tracking centers of mass of molecular selections
over trajectories, with special handling for membrane systems. Includes
automatic leaflet identification, z-coordinate filtering for selecting molecules
in specific leaflets, and optional tracking of membrane phosphate positions.
Uses NoJump transformation to handle periodic boundary crossings.
"""

import numpy as np
import MDAnalysis as mda
from lipyphilic.transformations import nojump
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
import MDAnalysis.transformations as trans
from MDAnalysis.transformations.nojump import NoJump
import pickle

def COM_sel(group_a, subdivision,**kwargs):
    """
    Calculate center of mass for a selection with optional filtering.
    
    Computes the center of mass for each fragment or residue in the
    selection, optionally filtering results using a boolean mask.
    
    Parameters
    ----------
    group_a : AtomGroup
        Selection of atoms for COM calculation.
    subdivision : str
        Level for COM calculation. Options:
        - "fragments": Calculate COM for each fragment
        - "residues": Calculate COM for each residue
    **kwargs : dict
        Optional keyword arguments:
        - mask : ndarray of bool, optional
            Boolean mask to filter results. If provided, only COMs
            where mask is True are returned.
            
    Returns
    -------
    x_total : list of ndarray
        List containing filtered COM coordinates.
        Shape: [1, n_filtered, 3] where n_filtered is the number
        of True values in mask (or total fragments/residues if no mask).
        
    Notes
    -----
    - Uses compound parameter in center_of_mass to group atoms
    - If mask is None, returns all COMs
    - If mask is provided, returns only COMs where mask[i] == True
    
    Examples
    --------
    >>> coms = COM_sel(peptides, 'fragments', mask=upper_leaflet_mask)
    >>> # Returns COMs only for fragments where mask is True
    """
    
    x_total=[]

    # Calculate COM for each fragment or residue
    x=group_a.center_of_mass(compound=f"{subdivision}")

    # Determine number of subdivisions
    if subdivision=="fragments":
        frag_no=group_a.fragments
    else:
        frag_no=group_a.residues

    # Apply mask if provided
    mask=kwargs.get('mask', None)
    x_pre=x[mask]
    x_total.append(x_pre)
    return x_total


def COM_phosphates(Upper_phosphates,Lower_phosphates):
    """
    Calculate center of mass for upper and lower leaflet phosphates.
    
    Computes the overall COM for phosphate groups in each leaflet,
    treating all atoms in each leaflet as a single group.
    
    Parameters
    ----------
    Upper_phosphates : AtomGroup
        Phosphate atoms in the upper leaflet.
    Lower_phosphates : AtomGroup
        Phosphate atoms in the lower leaflet.
        
    Returns
    -------
    Upper_positions : list of ndarray
        List containing upper leaflet phosphate COM. Shape: [1, 3]
    Lower_positions : list of ndarray
        List containing lower leaflet phosphate COM. Shape: [1, 3]
        
    Notes
    -----
    Uses compound='group' to treat all phosphates in each leaflet as
    a single group for COM calculation.
    
    
    """
    
    Upper_positions=[]
    Lower_positions=[]
    
    Upper_positions.append(Upper_phosphates.center_of_mass(compound='group'))
    Lower_positions.append(Lower_phosphates.center_of_mass(compound='group'))
    
    return Upper_positions,Lower_positions



class COM(AnalysisBase):
    
    """
    Track centers of mass for molecular selections over trajectory.
    
    This class calculates and stores center of mass coordinates for molecular
    selections (fragments or residues) over time. Includes automatic leaflet
    identification for membrane systems and filtering by z-coordinate to
    select molecules in specific leaflets. Uses NoJump transformation to
    handle molecule periodic boundary crossings.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    sel : str
        MDAnalysis selection string for molecules to track.
    phosphates : str, default="name PO1 PO2 PO4"
        MDAnalysis selection string for phosphate atoms used in
        leaflet identification.
    xy : bool, default=False
        If True, store xyz coordinates. If False with z=True, store
        only z-coordinates.
    z : bool, default=True
        If True, filter molecules by z-coordinate (leaflet selection).
        If False, track all molecules in selection.
    upper : bool, default=True
        If z=True: True selects upper leaflet molecules, False selects
        lower leaflet molecules. Upper leaflet defined as z > midplane
        and z < box_z + 5 Å.
    phos_dat : bool, default=False
        If True, also track phosphate COM positions for both leaflets.
    subdivision : str, default='fragments'
        Level for COM calculation. Options:
        - 'fragments': Calculate COM for each fragment
        - 'residues': Calculate COM for each residue
    mask_frame : int, default=499
        Frame index used to create the leaflet mask. Should be a
        representative equilibrated frame.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory with NoJump transformation applied.
    Selection : AtomGroup
        Selected atoms based on sel parameter.
    Upper_leaflet_phosphates : AtomGroup
        Phosphate atoms in the upper leaflet.
    Lower_leaflet_phosphates : AtomGroup
        Phosphate atoms in the lower leaflet.
    mask : ndarray of bool
        Boolean mask indicating which fragments/residues are in the
        selected leaflet (only if z=True).
    Selection_COMs : list
        List of COM positions at each frame. Structure depends on
        parameters (see Notes).
    Upper_phosphates : list of ndarray
        Upper leaflet phosphate COM at each frame (if phos_dat=True).
    Lower_phosphates : list of ndarray
        Lower leaflet phosphate COM at each frame (if phos_dat=True).
    
    Notes
    -----
    NoJump Transformation:
    - Automatically applied to prevent molecules from appearing to
      jump across periodic boundaries
    - Essential for tracking COM trajectories over time
    
    Leaflet Selection Logic (z=True):
    - Uses mask_frame to determine midplane position
    - Midplane = (upper_leaflet_z + lower_leaflet_z) / 2
    - Upper leaflet: midplane < z < box_z + 5
    - Lower leaflet: z < midplane OR z > box_z + 5
    - Mask created once at mask_frame and applied to all frames
    
    Output Structure:
    Selection_COMs structure depends on parameters:
    - z=True, xy=False: List of lists of z-coordinates
      [[z1, z2, ...], [z1, z2, ...], ...]  (n_frames lists)
    - z=True, xy=True: List of lists of [x,y,z] arrays
      [[[x,y,z], [x,y,z], ...], ...]  (n_frames lists)
    - z=False: List of lists of [x,y,z] arrays for all molecules
    
    Warnings
    --------
    - mask_frame should be chosen from equilibrated portion of trajectory
    - Upper/lower leaflet definition assumes membrane is centered in box
    - The "+ 5 Å" buffer in upper limit may need adjustment for curved membranes

    """
    
    def __init__(self,tpr,traj,sel,phosphates="name PO1 PO2 PO4",xy=False,z=True,upper=True,phos_dat=False,subdivision='fragments',mask_frame=499):
        universe=mda.Universe(tpr,traj)
        universe.trajectory.add_transformations(NoJump())
        super(COM,self).__init__(universe.trajectory)
        self.tpr=tpr
        self.traj=traj
        self.universe=universe
        self.phosphates=phosphates
        self.sel=sel
        self.subdivision=subdivision
        self.xy=xy
        self.z=z
        self.upper=upper
        
        # Whether to save phosphate COM positions or not
        self.phos_dat=phos_dat
        self.subdivision=subdivision
        self.mask_frame=mask_frame

        # Identify leaflets using LeafletFinder
        L=LeafletFinder(self.universe,self.phosphates,pbc="True")
        leaflet0=L.groups(0) # Upper leaflet
        leaflet1=L.groups(1) # Lower leaflet

        self.Selection=self.universe.select_atoms(self.sel)

        self.Upper_leaflet_phosphates=leaflet0.atoms
        self.Lower_leaflet_phosphates=leaflet1.atoms

        # Create leaflet mask if z-filtering is enabled
        if self.z==True:
            # Use mask_frame to determine which molecules are in target leaflet
            for ts in self.universe.trajectory[self.mask_frame:self.mask_frame+1]:
                COM_PMB1s_ref=self.Selection.center_of_mass(compound=self.subdivision)
                Upper_leaflet_positions=self.Upper_leaflet_phosphates.center_of_mass()
                Lower_leaflet_positions=self.Lower_leaflet_phosphates.center_of_mass()
                Middle_z=(Upper_leaflet_positions[2]+Lower_leaflet_positions[2])/2
                Upper_limit=self.universe.dimensions[2]+5

            # Create mask based on leaflet selection
            if self.upper==True:
                # Select molecules above midplane and below upper limit
                Mask=(COM_PMB1s_ref[:,2]>Middle_z) & (COM_PMB1s_ref[:,2]<Upper_limit) 
            elif self.upper==False:
                # Select molecules below midplane or above upper limit (lower leaflet)
                Mask=(COM_PMB1s_ref[:,2]<Middle_z) | (COM_PMB1s_ref[:,2]>Upper_limit)
            
            self.mask=Mask

    def _prepare(self):
        """
        Initialize storage lists before analysis.
        
        Called automatically by AnalysisBase.run() before iteration begins.
        Creates empty lists for storing COM data at each frame.
        """
        
        self.Selection_COMs=[]
        self.Upper_phosphates=[]
        self.Lower_phosphates=[]
       

    def _single_frame(self):
         """        
        Computes COMs for the selection (optionally filtered by leaflet mask)
        and optionally for phosphate groups. Storage format depends on
        parameters set during initialization.
        
        Called automatically by AnalysisBase.run() for each frame.
        
        Workflow
        --------
        If z=True (leaflet filtering enabled):
        1. Calculate COMs using mask to select specific leaflet
        2. Count number of molecules in mask
        3. If xy=True: Store full [x,y,z] coordinates
           If xy=False: Store only z-coordinates
        
        If z=False (no leaflet filtering):
        1. Calculate COMs for all molecules
        2. Store full [x,y,z] coordinates for all
        
        If phos_dat=True:
        - Calculate and store phosphate COMs for both leaflets
        
        Notes
        -----
        - Uses COM_sel helper function for mask-aware COM calculation
        - Mask is applied consistently across all frames
        - Phosphate tracking independent of main selection
        """

        if self.z==True:

             # Calculate COM with leaflet mask
            Selection_COM=COM_sel(group_a=self.Selection,subdivision=self.subdivision,mask=self.mask)

            # Count molecules in selected leaflet
            PMB1_no=sum((x==True) for x in self.mask)
        
            if self.xy==True:
                # Store full xyz coordinates
                Total_selection=[np.array(Selection_COM)[0,x,:] for x in range(PMB1_no)]
                self.Selection_COMs.append(Total_selection)

        
            elif self.xy==False:
                # Store only z-coordinates
                Total_selection=[np.array(Selection_COM)[0,x,2] for x in range(PMB1_no)]
                self.Selection_COMs.append(Total_selection)
               
        else:
            # No leaflet filtering: track all molecules
            Selection_COM=COM_sel(group_a=self.Selection,subdivision=self.subdivision) 

            # Determine number of fragments/residues
            if self.subdivision=="fragments":
                frag_no=len(self.Selection.fragments)
            else:
                frag_no=len(self.Selection.residues)
                
            # Store full xyz coordinates for all molecules
            Total_selection = [np.array(Selection_COM)[0,0,x,:] for x in range(frag_no)]
            self.Selection_COMs.append(Total_selection)

         # Track phosphate positions if requested
        if self.phos_dat!=None:
            Upper,Lower=COM_phosphates(self.Upper_leaflet_phosphates,self.Lower_leaflet_phosphates)
            self.Upper_phosphates.append(Upper)
            self.Lower_phosphates.append(Lower)
                
