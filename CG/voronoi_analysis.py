"""
Module for performing Voronoi tessellation on membrane systems, particularly useful for analyzing lipid packing and areas per lipid in coarse-grained simulations. Uses the freud library for efficient Voronoi calculations and handles protein-containing membrane systems.
"""

import freud
import numpy as np
import pickle 
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
import MDAnalysis.transformations as trans
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
from lipyphilic.transformations import nojump, triclinic_to_orthorhombic, center_membrane

def Protein_bead_selection(u,phosphate_sel,protein_sel,protein_id,cutoff1,cutoff2):
    """
    Select protein backbone beads near the membrane surface.
    
    Identifies protein backbone beads that are within a vertical range of
    nearby phosphate beads. This is useful for including relevant protein
    beads in Voronoi tessellation of the membrane surface.
    
    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe containing the membrane-protein system.
    phosphate_sel : str
        MDAnalysis selection string for phosphate atoms (e.g., "name PO4").
    protein_sel : str
        MDAnalysis selection string for protein (e.g., "protein").
    protein_id : str
        Selection string to identify a specific full protein instance
        (e.g., "resid 1-100" or "segid PROA").
    cutoff1 : float
        Minimum z-coordinate for initial protein selection (Å).
        Used to select proteins above a certain height.
    cutoff2 : float
        Vertical distance from average phosphate z-coordinate (Å).
        Defines the range [z_avg - cutoff2, z_avg + cutoff2] within which to
        select protein backbone beads.
        
    Returns
    -------
    Protein_BB : AtomGroup
        Protein backbone beads (name BB*) within the specified vertical
        range of nearby phosphates.
        
    Notes
    -----
    Workflow:
    1. Select phosphates within 10 Å of protein and above cutoff1
    2. Calculate average z-coordinate of these phosphates
    3. Define vertical range: [z_avg - cutoff2, z_avg + cutoff2]
    4. Select protein backbone beads (name BB*) within this range
    
    """
    
    #Select phosphates near protein and > z cutoff1
    Phosphates_near_protein=u.select_atoms(f"({phosphate_sel}) and around 10 (({protein_sel}) and ({protein_id}) and (prop z > {cutoff1}))",updating=True)
    
    #Calculate average phosphate z coordinate for the above phosphates.
    Protein_phosphate_ave=np.average(Phosphates_near_protein.positions[:,2])
    Lower_limit=Protein_phosphate_ave-cutoff2
    Upper_limit=Protein_phosphate_ave+cutoff2
    
    #Select pprotein backbone beads within z range defined by average z position of protein backbone beads and cutoff2.
    Protein_BB=u.select_atoms(f"name BB* and ({protein_sel}) and ({protein_id}) and (prop z > {Lower_limit}) and (prop z < {Upper_limit})")
    return Protein_BB

class Voronoi_analysis(AnalysisBase):
    """    
    This class generates Voronoi diagrams frame-by-frame for coarse-grained
    membrane systems. It identifies lipid phosphates in the upper leaflet,
    optionally includes protein backbone beads at the membrane surface, and
    computes Voronoi cells and volumes using the freud library.

    Adapted from Lipyphilic area per lipid code. 
    Inherits from MDAnalysis AnalysisBase class.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    Upper_PMB1s_sel : str, optional
        MDAnalysis selection string for PMB1 molecules in upper leaflet.
    Lower_PMB1s_sel : str, optional
        MDAnalysis selection string for PMB1 molecules in lower leaflet.
    phosphates : str, default="name PO1 PO2 PO4"
        MDAnalysis selection string for lipid phosphate atoms used in
        leaflet identification.
    Protein_sel : str, optional
        MDAnalysis selection string for protein atoms.
    Protein_ids : list of str, optional
        List of selection strings to identify individual proteins in the
        system. Required for protein-containing systems. Useful when
        individual proteins in ITP file are labeled from residue 1-x
        rather than sequentially (e.g., ["segid PROA", "segid PROB"]).
    cutoff1 : float, default=80
        Minimum z-coordinate (Å) for initial protein selection. Proteins
        above this height are considered for surface bead selection.
    cutoff2 : float, default=5
        Vertical distance (Å) from average phosphate z-coordinate for
        selecting protein backbone beads. Defines the range
        [z_avg ± cutoff2].
    **kwargs : dict
        Additional keyword arguments passed to AnalysisBase.__init__,
        such as start, stop, step, verbose.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    box_coords : list of float
        Box x-dimension at each frame.
    freud_coords : list of ndarray
        XY coordinates (with z=0) used for Voronoi tessellation at each frame.
        Shape: (3, n_points) per frame.
    freud_cells : list
        Voronoi polytopes (cells) at each frame from freud.locality.Voronoi.
    freud_volumes : list of ndarray
        Voronoi cell volumes (areas in 2D) at each frame.
    RAMP_phosphate_coords : list of ndarray
        Coordinates of RAMP phosphates (PO1, PO2) in upper leaflet at each frame.
    POPE_phosphate_coords : list of ndarray
        Coordinates of POPE phosphates (PO4) in upper leaflet at each frame.
    POPG_phosphate_coords : list of ndarray
        Coordinates of POPG phosphates (PO4) in upper leaflet at each frame.
    PMB1_upper_coords : list of ndarray
        Coordinates of upper PMB1 molecules at each frame (if provided).
    PMB1_lower_coords : list of ndarray
        Coordinates of lower PMB1 molecules at each frame (if provided).
    Protein_coords : list of ndarray
        Coordinates of all protein atoms at each frame (if provided).
    Protein_BB_coords : list of ndarray
        Coordinates of protein backbone beads at membrane surface at each
        frame (if protein present).
        
    
    Notes
    -----
    - Only upper leaflet lipids are included in tessellation
    - Uses freud.locality.Voronoi for efficient tessellation
    - Assumes square simulation box (freud.box.Box.square)

    """
    
   
    def __init__(self, tpr,traj,Upper_PMB1s_sel=None,Lower_PMB1s_sel=None,phosphates="name PO1 PO2 PO4",Protein_sel=None, Protein_ids=None,cutoff1=80, cutoff2=5,**kwargs):
        
        universe=mda.Universe(tpr,traj)
        super(Voronoi_analysis,self).__init__(universe.trajectory,**kwargs)
        self.tpr=tpr
        self.traj=traj
        self.universe=universe
        self.phosphates=phosphates
        self.Protein_sel=Protein_sel
        self.Upper_PMB1s_sel=Upper_PMB1s_sel
        self.Lower_PMB1s_sel=Lower_PMB1s_sel
        self.Protein_ids=Protein_ids
        self.cutoff1=cutoff1
        self.cutoff2=cutoff2


    def _prepare(self):
        """
        Initialize data storage lists before analysis.
        
        Called automatically by AnalysisBase.run() before iteration begins.
        Creates empty lists for storing coordinates, Voronoi cells, and
        volumes at each frame.
        """
        self.box_coords=[]
        self.freud_coords=[]
        self.freud_cells=[]
        self.freud_volumes=[]
        self.RAMP_phosphate_coords=[]
        self.POPE_phosphate_coords=[]
        self.POPG_phosphate_coords=[]
        self.PMB1_upper_coords=[]
        self.PMB1_lower_coords=[]
        self.Protein_coords=[]
        self.Protein_BB_coords=[]

    def _single_frame(self):
        """
        Perform Voronoi tessellation for current frame.
        
        Workflow:
        1. Identify upper leaflet using LeafletFinder (LeafletFinder.groups(0))
        2. Extract and store phosphate coordinates by lipid type
        3. Create freud Box object from simulation dimensions (with z set to 0)
        4. Extract and store PMB1 coordinates (if present)
        5. For protein systems:
           a) Select protein backbone beads at membrane surface
           b) Combine phosphates and protein beads for tessellation
           c) Store protein coordinates
        6. Perform Voronoi tessellation in 2D (z=0 plane)
        7. Store polytopes and volumes
        
        Called automatically by AnalysisBase.run() for each frame.
        
        Raises
        ------
        May raise errors if Protein_sel is provided but Protein_ids is None.
        """
        
        # Identify upper leaflet phosphates
        L=LeafletFinder(self.universe,self.phosphates,pbc=True)
        leaflet0=L.groups(0)

        # Store phosphate coordinates
        RAMP_phosphates_upper=leaflet0.select_atoms('name PO1 PO2')
        POPE_phosphates_upper=leaflet0.select_atoms('resname POPE and name PO4')
        POPG_phosphates_upper=leaflet0.select_atoms('resname POPG and name PO4')
        self.RAMP_phosphate_coords.append(RAMP_phosphates_upper.positions)
        self.POPE_phosphate_coords.append(POPE_phosphates_upper.positions)
        self.POPG_phosphate_coords.append(POPG_phosphates_upper.positions)

        # Store box dimensions and create freud.box.Box object
        self.box_coords.append(self.universe.dimensions[0])
        Box_Freud=freud.box.Box.square(self.universe.dimensions[0])

        # Handle PMB1 peptides if present
        if self.Upper_PMB1s_sel!=None:
                PMB1s_upper=self.universe.select_atoms(self.Upper_PMB1s_sel)
                self.PMB1_upper_coords.append(PMB1s_upper.positions)
        
        if self.Lower_PMB1s_sel!=None:
                PMB1s_lower=self.universe.select_atoms(self.Lower_PMB1s_sel)
                self.PMB1_lower_coords.append(PMB1s_lower.positions)

        # Handle protein if present
        if self.Protein_sel!=None:
            Protein=self.universe.select_atoms(self.Protein_sel)
            self.Protein_coords.append(Protein.positions)
            Protein_BB_list=[]

            # Select backbone beads for each protein
            for i in self.Protein_ids:
                Protein_BB=Protein_bead_selection(u=self.universe,phosphate_sel=self.phosphates, protein_sel=self.Protein_sel,protein_id=i,cutoff1=self.cutoff1,cutoff2=self.cutoff2)
                Protein_BB_list.append(Protein_BB)

            # Start with leaflet phosphates    
            Protein_phosphates=leaflet0

            # Create empty AtomGroup for protein backbone beads
            x = self.universe.select_atoms("resname DOES_NOT_EXIST")
            Protein_BB_full=x

            # Append AtomGroups in Protein_BB_list to Protein_phosphates and Protein_BB_full
            for i in Protein_BB_list:
                Protein_phosphates+=i
                Protein_BB_full+=i

            # Store protein backbone coordinates
            self.Protein_BB_coords.append(Protein_BB_full.positions)

            # Create 2D points for Voronoi tessellation (x, y coordinates of Protein_phosphates)
            zeroes=np.zeros(len(Protein_phosphates))
            Points=np.vstack(([Protein_phosphates.positions[:,0],Protein_phosphates.positions[:,1],zeroes]))
            self.freud_coords.append(Points)

            # Compute Voronoi tessellation
            voro=freud.locality.Voronoi(verbose=True)

            self.freud_cells.append(voro.compute((Box_Freud,np.transpose(Points))).polytopes)
            self.freud_volumes.append(voro.compute((Box_Freud,np.transpose(Points))).volumes)


        else:
            # No protein - use only phosphates
            Protein_coords=[0,0]
            Protein_BB_coords=[]

             # Create 2D points for Voronoi tessellation (x, y coordinates of upper leaflet phosphates)
            Points=np.vstack([leaflet0.positions[:,0],leaflet0.positions[:,1],zeroes])
            self.freud_coords.append(Points)

            # Compute Voronoi tesselation
            voro=freud.locality.Voronoi(verbose=True)
            self.freud_cells.append(voro.compute((Box_Freud,np.transpose(Points))).polytopes)
            self.freud_volumes.append(voro.compute((Box_Freud,np.transpose(Points))).volumes)

       

    def _conclude(self):
        """
        Finalize analysis after all frames processed.
        
        Called automatically by AnalysisBase.run() after processing all frames.
        Currently performs no operations (normalization/averaging could be
        added here if needed).
        """
        None
        

                         
        
    
