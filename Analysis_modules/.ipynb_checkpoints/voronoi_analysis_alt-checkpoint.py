import freud
import numpy as np
import pickle 
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
import MDAnalysis.transformations as trans
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.base import (AnalysisBase,
                                      AnalysisFromFunction,
                                      analysis_class)
from lipyphilic.transformations import nojump, triclinic_to_orthorhombic, center_membrane

def pickle_files(filename,variable):
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

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

def Time_positions(u,Upper_PMB1s_sel,Lower_PMB1s_sel,RAMP_phosphates,POPE_phosphates,POPG_phosphates,Upper_leaflet,Protein_sel):
    box_coords=[]
    freud_coords=[]
    freud_cells=[]
    freud_volumes=[]
    RAMP_phosphate_coords=[]
    POPE_phosphate_coords=[]
    POPG_phosphate_coords=[]
    PMB1_upper_coords=[]
    PMB1_lower_coords=[]
    Protein_coords=[]
    
    if Upper_PMB1s_sel!=None:
        PMB1s_upper=u.select_atoms(Upper_PMB1s_sel)
        PMB1_upper_coords.append(PMB1s_upper.positions)
    else:
        PMB1s_upper_coords=[0,0]
    if Lower_PMB1s_sel!=None:
        PMB1s_lower=u.select_atoms(Lower_PMB1s_sel)
        PMB1_lower_coords.append(PMB1s_lower.positions)
    else:
        PMB1s_lower_coords=[0,0]
    
    if Protein_sel !=None:
        Proteins=u.select_atoms(Protein_sel)
        Protein_coords.append(Proteins.positions)
    else:
        Protein_coords=[0,0]
        
    box_coords.append(u.dimensions[0])
    
    zeroes=np.zeros(len(Upper_leaflet))
    
    Points=np.vstack(([Upper_leaflet.positions[:,0],Upper_leaflet.positions[:,1],zeroes]))
    freud_coords.append(Points)
    
    RAMP_phosphate_coords.append(RAMP_phosphates.positions)
    POPE_phosphate_coords.append(POPE_phosphates.positions)
    POPG_phosphate_coords.append(POPG_phosphates.positions)
        
    Box_Freud=freud.box.Box.square(u.dimensions[0])
    voro=freud.locality.Voronoi(verbose=True)
    freud_cells.append(voro.compute((Box_Freud,np.transpose(Points))).polytopes)
    freud_volumes.append(voro.compute((Box_Freud,np.transpose(Points))).volumes)
    
    return box_coords,freud_coords, freud_volumes, freud_cells,RAMP_phosphate_coords,POPE_phosphate_coords,POPG_phosphate_coords,PMB1_upper_coords,PMB1_lower_coords,Protein_coords
                         
Calculate_Time_positions = analysis_class(Time_positions)

class Voronoi_diagram():
    def __init__(self,tpr,traj,saving_folder,Upper_PMB1s_sel,Lower_PMB1s_sel,phosphates="name PO1 PO2 PO4",start=0,stop=-1,step=1,Protein_sel=None,verbose=True, Protein_ids=None, cutoff1=80, cutoff2=5):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj)
        self.phosphates=phosphates
        self.start=start
        self.stop=stop
        self.step=step
        self.Protein_sel=Protein_sel
        self.verbose=verbose
        self.savingfolder=saving_folder
        self.Upper_PMB1s_sel=Upper_PMB1s_sel
        self.Lower_PMB1s_sel=Lower_PMB1s_sel
        self.Protein_ids=Protein_ids
        self.cutoff1=cutoff1
        self.cutoff2=cutoff2
        
    def Voronoi_analysis(self):
        L=LeafletFinder(self.universe,self.phosphates,pbc=True)
        leaflet0=L.groups(0)        
        
        RAMP_phosphates_upper=leaflet0.select_atoms('name PO1 PO2')
        POPE_phosphates_upper=leaflet0.select_atoms('resname POPE and name PO4')
        POPG_phosphates_upper=leaflet0.select_atoms('resname POPG and name PO4')
        
        
        if self.Protein_sel!=None:
            Protein=self.universe.select_atoms(self.Protein_sel)
            Protein_BB_list=[]
            for i in self.Protein_ids:
                Protein_BB=Protein_bead_selection(u=self.universe,phosphate_sel=self.phosphates, protein_sel=self.Protein_sel,protein_id=i,cutoff1=self.cutoff1,cutoff2=self.cutoff2)
                Protein_BB_list.append(Protein_BB)
                
            Protein_phosphates=leaflet0
            for i in Protein_BB_list:
                Protein_phosphates+i
            p=Calculate_Time_positions(u=self.universe,Upper_PMB1s_sel=self.Upper_PMB1s_sel,Lower_PMB1s_sel=self.Lower_PMB1s_sel,POPE_phosphates=POPE_phosphates_upper,POPG_phosphates=POPG_phosphates_upper,RAMP_phosphates=RAMP_phosphates_upper,Upper_leaflet=Protein_phosphates,Protein_sel=self.Protein_sel)

        
        else:
            p=Calculate_Time_positions(u=self.universe,Upper_PMB1s_sel=self.Upper_PMB1s_sel,Lower_PMB1s_sel=self.Lower_PMB1s_sel, POPE_phosphates=POPE_phosphates_upper,POPG_phosphates=POPG_phosphates_upper,RAMP_phosphates=RAMP_phosphates_upper,Upper_leaflet=leaflet0,Protein_sel=self.Protein_sel)
        
        p.run(start=self.start,stop=self.stop,verbose=self.verbose)            
        
        
        Freud_volumes=p.results["timeseries"][0][2,0]
        Freud_cells=p.results["timeseries"][0][3,0]
        LPS_phosphate_coords_x=np.array(p.results["timeseries"][0][4,0])[:,0]
        LPS_phosphate_coords_y=np.array(p.results["timeseries"][0][4,0])[:,1]
        
        
        if self.Upper_PMB1s_sel != None:
            Upper_PMB1_coords_x=np.array(p.results["timeseries"][0][7,0])[:,0]
            Upper_PMB1_coords_y=np.array(p.results["timeseries"][0][7,0])[:,1]
        else:
            Upper_PMB1_coords_x=0
            Upper_PMB1_coords_y=0
        
        if self.Lower_PMB1s_sel != None:
            Lower_PMB1_coords_x=np.array(p.results["timeseries"][0][8,0])[:,0]
            Lower_PMB1_coords_y=np.array(p.results["timeseries"][0][8,0])[:,1]
        else:
            Lower_PMB1_coords_x=0
            Lower_PMB1_coords_y=0
            
        if self.Protein_sel != None:
            Protein_coords_x=np.array(p.results["timeseries"][0][9,0])[:,0]
            Protein_coords_y=np.array(p.results["timeseries"][0][9,0])[:,1]
        else:
            Protein_coords_x=0
            Protein_coords_y=0

        
        pickle_files(f"{self.savingfolder}Protein_coords",[Protein_coords_x,Protein_coords_y]) 
        pickle_files(f"{self.savingfolder}Voronoi_volumes",Freud_volumes)
        pickle_files(f"{self.savingfolder}Voronoi_cells",Freud_cells)
        pickle_files(f"{self.savingfolder}LPS_phosphate_coords",[LPS_phosphate_coords_x, LPS_phosphate_coords_y])
        pickle_files(f"{self.savingfolder}Upper_PMB1_coords",[Upper_PMB1_coords_x, Upper_PMB1_coords_y])
        pickle_files(f"{self.savingfolder}Lower_PMB1_coords",[Lower_PMB1_coords_x,Lower_PMB1_coords_y])
                        
        return Freud_volumes, Freud_cells, [LPS_phosphate_coords_x, LPS_phosphate_coords_y], [Upper_PMB1_coords_x, Upper_PMB1_coords_y],[Lower_PMB1_coords_x,Lower_PMB1_coords_y],[Protein_coords_x,Protein_coords_y]
        
        
        
                         
       

                         
        
    
