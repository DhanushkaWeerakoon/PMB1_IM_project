"""
Module to calculate number of charged species in interior and exterior regions of double bilayer membrane systems.

"""


import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)


class Electroporation_check(AnalysisBase):
    """    
    This class tracks the number of ions (Na+, Cl-, Ca2+), charged lipids
    (POPG), and charged amino acid sidechains (Glu, Asp, Arg, Lys) in two
    regions of double membrane systems: 
    (1) between the membranes (central/interior), and
    (2) outside the membranes (outer/exterior). 

    Inherits from MDAnalysis.AnalysisBase class.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    midpoint : float
        Initial z-coordinate (Å) of midpoint of bilayer system. Used for initial leaflet
        identification.
    polarisable : bool, default=False
        Force field type for amino acid selection:
        - False: Standard MARTINI (SC1 for Glu/Asp, SC2 for Arg/Lys)
        - True: Polarizable MARTINI (SCN for Glu/Asp, SCP for Arg/Lys)
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    Upper_bilayer : AtomGroup
        Phosphate atoms (PO1, PO2, PO4) in the upper leaflet.
    Lower_bilayer : AtomGroup
        Phosphate atoms (PO1, PO2, PO4) in the lower leaflet.
    Bilayer_COMs : list of float
        Bilayer separation (upper_COM_z - lower_COM_z) at each frame (Å).
    Central_NAs : list of int
        Number of Na+ ions between membranes at each frame.
    Central_CLs : list of int
        Number of Cl- ions between membranes at each frame.
    Central_CAs : list of int
        Number of Ca2+ ions between membranes at each frame.
    Central_POPG : list of int
        Number of POPG lipids between membranes at each frame.
    Central_Glu : list of int
        Number of Glu sidechains between membranes at each frame.
    Central_Asp : list of int
        Number of Asp sidechains between membranes at each frame.
    Central_Arg : list of int
        Number of Arg sidechains between membranes at each frame.
    Central_Lys : list of int
        Number of Lys sidechains between membranes at each frame.
    Outer_NAs : list of int
        Number of Na+ ions outside membranes at each frame.
    Outer_CLs : list of int
        Number of Cl- ions outside membranes at each frame.
    Outer_CAs : list of int
        Number of Ca2+ ions outside membranes at each frame.
    Outer_POPG : list of int
        Number of POPG lipids outside membranes at each frame.
    Outer_Glu : list of int
        Number of Glu sidechains outside membranes at each frame.
    Outer_Asp : list of int
        Number of Asp sidechains outside membranes at each frame.
    Outer_Arg : list of int
        Number of Arg sidechains outside membranes at each frame.
    Outer_Lys : list of int
        Number of Lys sidechains outside membranes at each frame.
        
    
    Notes
    -----
    Region Definitions:
    - Central (interior): l_COM_z < z < u_COM_z
    - Outer (exterior): z > u_COM_z OR z < l_COM_z
    - Boundaries dynamically updated each frame based on phosphate COMs
    
    Tracked Species:
    Ions: Na+ (+1), Cl- (-1), Ca2+ (+2)
    Lipids: POPG (anionic, -1)
    Amino acids: Glu (-1), Asp (-1), Arg (+1), Lys (+1)
    
    """
    
    def __init__(self,tpr,traj,midpoint,polarisable=False):
        u=mda.Universe(tpr,traj,continuous=True)
        super(Electroporation_check,self).__init__(u.trajectory)
        self.tpr=tpr
        self.traj=traj
        self.universe=u
        self.midpoint=midpoint
        self.polarisable=polarisable

    def _prepare(self):
        """
        Initialize leaflet selections and storage lists before analysis.
        
        Called automatically by AnalysisBase.run() before iteration begins.
        Identifies upper and lower leaflet phosphates based on the midpoint
        parameter and creates empty lists for storing counts at each frame.
        
        Notes
        -----
        - Upper leaflet: phosphates with z > midpoint
        - Lower leaflet: phosphates with z < midpoint
        - Prints number of phosphates in each leaflet for verification
        - Creates 16 storage lists (8 central + 8 outer species)
        """
        
        # Identify leaflets based on initial midpoint
        self.Upper_bilayer=self.universe.select_atoms(f"name PO1 PO2 PO4 and (prop z > {self.midpoint})")
        self.Lower_bilayer=self.universe.select_atoms(f"name PO1 PO2 PO4 and (prop z < {self.midpoint})")

        
        # Initialize storage for central (interior) species
        self.Central_NAs=[]
        self.Central_CLs=[]
        self.Central_CAs=[]
        self.Central_POPG=[]
        self.Central_Glu=[]
        self.Central_Asp=[]
        self.Central_Arg=[]
        self.Central_Lys=[]

        # Initialize storage for outer (exterior) species
        self.Outer_NAs=[]
        self.Outer_CLs=[]
        self.Outer_CAs=[]
        self.Outer_POPG=[]
        self.Outer_Glu=[]
        self.Outer_Asp=[]
        self.Outer_Arg=[]
        self.Outer_Lys=[]

        # Initialize storage for bilayer separation
        self.Bilayer_COMs=[]
        

    def _single_frame(self):
        """
        Count charged species in central and outer regions for current frame.
        
        Calculates the center of mass z-coordinates for upper and lower
        leaflet phosphates, then counts ions, charged lipids, and charged
        amino acids in the central region (between membranes) and outer
        regions (outside membranes).
        
        Called automatically by AnalysisBase.run() for each frame.
        
        Workflow
        --------
        1. Calculate upper and lower leaflet phosphate COM z-coordinates
        2. Compute bilayer separation (u_COM_z - l_COM_z)
        3. For each species type:
           a) Select atoms in central region (l_COM_z < z < u_COM_z)
           b) Select atoms in outer regions (z > u_COM_z OR z < l_COM_z)
           c) Count and store number of atoms
        4. Amino acid selections depend on polarisable parameter
        
        Notes
        -----
   
        Species Selection Criteria:
        Ions:
        - Na+: name NA
        - Cl-: name CL
        - Ca2+: name CA
        
        Lipids:
        - POPG: resname POPG and name PO4
        
        Amino acids (standard MARTINI):
        - Glu: resname GLU and name SC1
        - Asp: resname ASP and name SC1
        - Arg: resname ARG and name SC2
        - Lys: resname LYS and name SC2
        
        Amino acids (polarizable MARTINI):
        - Glu: resname GLU and name SCN
        - Asp: resname ASP and name SCN
        - Arg: resname ARG and name SCP
        - Lys: resname LYS and name SCP
   
        """
        # Calculate leaflet COM z-coordinates
        u_COM_z=self.Upper_bilayer.center_of_mass(compound='group')[2]
        l_COM_z=self.Lower_bilayer.center_of_mass(compound='group')[2]

        # Calculate and store bilayer separation
        Bilayer_separation=u_COM_z-l_COM_z
        self.Bilayer_COMs.append(Bilayer_separation)

        # Select ions and lipids in central region
        Central_NA=self.universe.select_atoms(f"name NA and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
        Central_CL=self.universe.select_atoms(f"name CL and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
        Central_CA=self.universe.select_atoms(f"name CA and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
        Central_POPG=self.universe.select_atoms(f"(resname POPG and name PO4) and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")


        # Select ions and lipids in outer regions
        Outer_NA=self.universe.select_atoms(f"name NA and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
        Outer_CL=self.universe.select_atoms(f"name CL and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
        Outer_CA=self.universe.select_atoms(f"name CA and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
        Outer_POPG=self.universe.select_atoms(f"(resname POPG and name PO4) and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")

        # Select amino acids based on force field type
        if self.polarisable == False:
            # Standard MARTINI amino acid selections
            Central_Glu=self.universe.select_atoms(f"(resname GLU and name SC1) and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
            Central_Asp=self.universe.select_atoms(f"(resname ASP and name SC1) and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
            Central_Arg=self.universe.select_atoms(f"(resname ARG and name SC2) and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
            Central_Lys=self.universe.select_atoms(f"(resname LYS and name SC2) and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
            Outer_Glu=self.universe.select_atoms(f"(resname GLU and name SC1) and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
            Outer_Asp=self.universe.select_atoms(f"(resname ASP and name SC1) and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
            Outer_Arg=self.universe.select_atoms(f"(resname ARG and name SC2) and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
            Outer_Lys=self.universe.select_atoms(f"(resname LYS and name SC2) and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")

        if self.polarisable == True:
            # Polarizable MARTINI amino acid selections
            Central_Glu=self.universe.select_atoms(f"(resname GLU and name SCN) and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
            Central_Asp=self.universe.select_atoms(f"(resname ASP and name SCN) and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
            Central_Arg=self.universe.select_atoms(f"(resname ARG and name SCP) and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
            Central_Lys=self.universe.select_atoms(f"(resname LYS and name SCP) and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
            Outer_Glu=self.universe.select_atoms(f"(resname GLU and name SCN) and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
            Outer_Asp=self.universe.select_atoms(f"(resname ASP and name SCN) and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
            Outer_Arg=self.universe.select_atoms(f"(resname ARG and name SCP) and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
            Outer_Lys=self.universe.select_atoms(f"(resname LYS and name SCP) and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")

        # Store counts for central species
        self.Central_NAs.append(len(Central_NA))
        self.Central_CLs.append(len(Central_CL))
        self.Central_CAs.append(len(Central_CA))
        self.Central_POPG.append(len(Central_POPG))
        self.Central_Glu.append(len(Central_Glu))
        self.Central_Asp.append(len(Central_Asp))
        self.Central_Arg.append(len(Central_Arg))
        self.Central_Lys.append(len(Central_Lys))

        # Store counts for outer species
        self.Outer_NAs.append(len(Outer_NA))
        self.Outer_CLs.append(len(Outer_CL))
        self.Outer_CAs.append(len(Outer_CA))
        self.Outer_POPG.append(len(Outer_POPG))
        self.Outer_Glu.append(len(Outer_Glu))
        self.Outer_Asp.append(len(Outer_Asp))
        self.Outer_Arg.append(len(Outer_Arg))
        self.Outer_Lys.append(len(Outer_Lys))



