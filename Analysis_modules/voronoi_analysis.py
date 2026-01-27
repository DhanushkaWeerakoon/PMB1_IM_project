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
    #Select phosphates surrounding proteins
    Phosphates_near_protein=u.select_atoms(f"({phosphate_sel}) and around 10 (({protein_sel}) and ({protein_id}) and (prop z > {cutoff1}))",updating=True)
    #Determine protein atoms within radius of these phosphates
    Protein_phosphate_ave=np.average(Phosphates_near_protein.positions[:,2])
    Lower_limit=Protein_phosphate_ave-cutoff2
    Upper_limit=Protein_phosphate_ave+cutoff2
    #Determine protein atoms within radius of those phosphates
    Protein_BB=u.select_atoms(f"name BB* and ({protein_sel}) and ({protein_id}) and (prop z > {Lower_limit}) and (prop z < {Upper_limit})")
    return Protein_BB

class Voronoi_analysis(AnalysisBase):
    """ 
    Class to generate Voronoi diagrams on a frame-by-frame basis for CG systems
    """
    def __init__(self, tpr,traj,Upper_PMB1s_sel=None,Lower_PMB1s_sel=None,phosphates="name PO1 PO2 PO4",Protein_sel=None, Protein_ids=None,cutoff1=80, cutoff2=5,**kwargs):
        """
        tpr: tpr file - required
        traj: trajectory file - required
        phosphates: lipid phosphate MDAnalysis selection string - default: "name PO1 PO2 PO4"
        Protein_sel: protein MDAnalysis selection string - default: None
        Upper_PMB1s_sel: MDAnalysis selection string for PMB1 molecules in the upper leaflet - default: None
        Lower_PMB1s_sel: MDAnalysis selection string for PMB1 molecules in the upper leaflet - default: None
        Protein_ids: MDAnalysis string to identify each individual protein in the system. Useful when individual proteins in ITP file are labelled from residue 1-x, as opposed to being labelled sequentially. Currently required for protein-containing systems.
        cutoff1
        cutoff2
        """
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
        # OPTIONAL
        # Called before iteration on the trajectory has begun.
        # Data structures can be set up at this time
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
        1. Identifies phosphates in upper leaflet. Saves phosphate coordinates.
        2. Creates Freud box and saves box coordinates.
        3. Saves PMB1 coordinates if present
        4. If protein is present - determines beads at membrane surface close to phosphate beads and includes them in bead selection used for Voronoi tesselation. Also save protein coordinates.
        5. Do Voronoi tesselation using Freud functionalities
        
        """
        L=LeafletFinder(self.universe,self.phosphates,pbc=True)
        leaflet0=L.groups(0)
        RAMP_phosphates_upper=leaflet0.select_atoms('name PO1 PO2')
        POPE_phosphates_upper=leaflet0.select_atoms('resname POPE and name PO4')
        POPG_phosphates_upper=leaflet0.select_atoms('resname POPG and name PO4')
        self.RAMP_phosphate_coords.append(RAMP_phosphates_upper.positions)
        self.POPE_phosphate_coords.append(POPE_phosphates_upper.positions)
        self.POPG_phosphate_coords.append(POPG_phosphates_upper.positions)

        self.box_coords.append(self.universe.dimensions[0])
        Box_Freud=freud.box.Box.square(self.universe.dimensions[0])
        
        if self.Upper_PMB1s_sel!=None:
                PMB1s_upper=u.select_atoms(Upper_PMB1s_sel)
                self.PMB1_upper_coords.append(PMB1s_upper.positions)
        
        if self.Lower_PMB1s_sel!=None:
                PMB1s_lower=u.select_atoms(Lower_PMB1s_sel)
                self.PMB1_lower_coords.append(PMB1s_lower.positions)
        
        if self.Protein_sel!=None:
            Protein=self.universe.select_atoms(self.Protein_sel)
            self.Protein_coords.append(Protein.positions)
            Protein_BB_list=[]

            for i in self.Protein_ids:
                Protein_BB=Protein_bead_selection(u=self.universe,phosphate_sel=self.phosphates, protein_sel=self.Protein_sel,protein_id=i,cutoff1=self.cutoff1,cutoff2=self.cutoff2)
                Protein_BB_list.append(Protein_BB)
                
            Protein_phosphates=leaflet0

            # Create an empty atomgroup and set it equal to Protein_BB_full 
            x = self.universe.select_atoms("resname DOES_NOT_EXIST")
            Protein_BB_full=x
            for i in Protein_BB_list:
                # Append the atomselections in Protein_BB_list to atomselections Protein_phosphates and Protein_BB_full
                Protein_phosphates+=i
                Protein_BB_full+=i

            # Work out what to do hre - what is Protein_BB_sel?
            self.Protein_BB_coords.append(Protein_BB_full.positions)

            zeroes=np.zeros(len(Protein_phosphates))
            Points=np.vstack(([Protein_phosphates.positions[:,0],Protein_phosphates.positions[:,1],zeroes]))
            self.freud_coords.append(Points)
            voro=freud.locality.Voronoi(verbose=True)

            self.freud_cells.append(voro.compute((Box_Freud,np.transpose(Points))).polytopes)
            self.freud_volumes.append(voro.compute((Box_Freud,np.transpose(Points))).volumes)


        else:
            Protein_coords=[0,0]
            Protein_BB_coords=[]

            Points=np.vstack([leaflet0.positions[:,0],leaflet0.positions[:,1],zeroes])
            self.freud_coords.append(Points)
            voro=freud.locality.Voronoi(verbose=True)
            self.freud_cells.append(voro.compute((Box_Freud,np.transpose(Points))).polytopes)
            self.freud_volumes.append(voro.compute((Box_Freud,np.transpose(Points))).volumes)

       

    def _conclude(self):
        None
        # OPTIONAL
        # Called once iteration on the trajectory is finished.
        # Apply normalisation and averaging to results here.
       

                         
        
    
