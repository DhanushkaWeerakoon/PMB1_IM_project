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

def Protein_bead_selection(u,phosphate_sel,protein_sel,cutoff1,cutoff2):
    #Select phosphates surrounding proteins
    Phosphates_near_protein=u.select_atoms(f"({phosphate_sel}) and around 10 (({protein_sel}) and (prop z > {cutoff1}))",updating=True)
    #Determine protein atoms within radius of these phosphates
    Protein_phosphate_ave=np.average(Phosphates_near_protein.positions[:,2])
    Lower_limit=Protein_phosphate_ave-cutoff2
    Upper_limit=Protein_phosphate_ave+cutoff2
    print(Protein_phosphate_ave)
    #Determine protein atoms within radius of those phosphates
    Protein_BB=u.select_atoms(f"name BB* and ({protein_sel}) and (prop z > {Lower_limit}) and (prop z < {Upper_limit})")
    return Protein_BB

def Time_positions(u,RAMP_phosphates,POPE_phosphates,POPG_phosphates,Upper_leaflet,Proteins,Protein_sel):
    box_coords=[]
    freud_coords=[]
    freud_cells=[]
    freud_volumes=[]
    RAMP_phosphate_coords=[]
    POPE_phosphate_coords=[]
    POPG_phosphate_coords=[]
    Protein_coords=[]
        
    box_coords.append(u.dimensions[0])
    
    zeroes=np.zeros(len(Upper_leaflet))
    
    Points=np.vstack(([Upper_leaflet.positions[:,0],Upper_leaflet.positions[:,1],zeroes]))
    freud_coords.append(Points)
    
    RAMP_phosphate_coords.append(RAMP_phosphates.positions)
    POPE_phosphate_coords.append(POPE_phosphates.positions)
    POPG_phosphate_coords.append(POPG_phosphates.positions)

    
    if Protein_sel != None:
        Protein_coords.append(Proteins.positions)
        
    Box_Freud=freud.box.Box.square(u.dimensions[0])
    voro=freud.locality.Voronoi(verbose=True)
    freud_cells.append(voro.compute((Box_Freud,np.transpose(Points))).polytopes)
    freud_volumes.append(voro.compute((Box_Freud,np.transpose(Points))).volumes)
    
    return box_coords,freud_coords, freud_volumes, freud_cells,RAMP_phosphate_coords,POPE_phosphate_coords,POPG_phosphate_coords,Protein_coords
                         
Calculate_Time_positions = analysis_class(Time_positions)

class Voronoi_diagram():
    def __init__(self,tpr,traj,saving_folder,phosphates="name PO1 PO2 PO4",start=0,stop=-1,step=1,snapshot_no=5,Protein_sel=None,verbose=True,cutoff1=80,cutoff2=5):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj)
        self.phosphates=phosphates
        self.start=start
        self.stop=stop
        self.step=step
        self.snapshot_no=snapshot_no
        self.Protein_sel=Protein_sel
        self.verbose=verbose
        self.savingfolder=saving_folder
        self.cutoff1=cutoff1
        self.cutoff2=cutoff2
        
    def Voronoi_analysis(self):
        L=LeafletFinder(self.universe,self.phosphates,pbc=True)
        leaflet0=L.groups(0)        
        
        RAMP_phosphates_upper=leaflet0.select_atoms('name PO1 PO2')
        POPE_phosphates_upper=leaflet0.select_atoms('resname POPE and name PO4')
        POPG_phosphates_upper=leaflet0.select_atoms('resname POPG and name PO4')        
        
        z=int((self.stop-self.start)/5)
        steps=[]
        for i in range(self.snapshot_no+1):
            x=self.start
            x+=(i*z)
            steps.append(x)
        
        if self.Protein_sel!=None:
            Protein=self.universe.select_atoms(self.Protein_sel)
            Protein_BB=Protein_bead_selection(u=self.universe,phosphate_sel=self.phosphates,protein_sel=self.Protein_sel,cutoff1=self.cutoff1,cutoff2=self.cutoff2)
            
            Protein_phosphates=leaflet0+Protein_BB
            
            Protein_BB_coords=Protein_BB.positions
            
            Data_master=[]
            for i in steps:
                p=Calculate_Time_positions(u=self.universe,POPE_phosphates=POPE_phosphates_upper,POPG_phosphates=POPG_phosphates_upper,RAMP_phosphates=RAMP_phosphates_upper,Upper_leaflet=Protein_phosphates,Proteins=Protein,Protein_sel=self.Protein_sel)
                p.run(start=i,stop=i+1,step=self.step,verbose=self.verbose)
                Data_master.append(p)
            
            print(np.shape(Data_master[0].results["timeseries"][0,4,0]))
            Freud_volumes=[Data_master[i].results["timeseries"][0,2,0] for i in range(self.snapshot_no+1)]
            Freud_cells=[Data_master[i].results["timeseries"][0,3,0] for i in range(self.snapshot_no+1)]
            LPS_phosphate_coords_x=[np.array(Data_master[i].results["timeseries"][0,4,0])[:,0] for i in range(self.snapshot_no+1)]
            LPS_phosphate_coords_y=[np.array(Data_master[i].results["timeseries"][0,4,0])[:,1] for i in range(self.snapshot_no+1)] 
            Protein_coords_x=[np.array(Data_master[i].results["timeseries"][0,7,0])[:,0] for i in range(self.snapshot_no+1)]
            Protein_coords_y=[np.array(Data_master[i].results["timeseries"][0,7,0])[:,1] for i in range(self.snapshot_no+1)]  
            pickle_files(f"{self.savingfolder}Protein_coords",[Protein_coords_x,Protein_coords_y])
            
                                               
        else:
            Protein=None
            Protein_BB_coords=None
                    
            Data_master=[]
            for i in steps:
                p=Calculate_Time_positions(u=self.universe,Upper_PMB1s=PMB1s_upper,Lower_PMB1s=PMB1s_lower,POPE_phosphates=POPE_phosphates_upper,POPG_phosphates=POPG_phosphates_upper,RAMP_phosphates=RAMP_phosphates_upper,Upper_leaflet=leaflet0,Proteins=Protein,Protein_sel=self.Protein_sel)
                p.run(start=i,stop=i+1,step=self.step,verbose=self.verbose)
                Data_master.append(p)
                

            print(np.shape(Data_master[0].results["timeseries"]))
        
            Freud_volumes=[Data_master[i].results["timeseries"][0,2] for i in range(self.snapshot_no+1)]
            Freud_cells=[Data_master[i].results["timeseries"][0,3] for i in range(self.snapshot_no+1)]
            LPS_phosphate_coords_x=[np.array(Data_master[i].results["timeseries"][0,4])[0,:,0] for i in range(self.snapshot_no+1)]
            LPS_phosphate_coords_y=[np.array(Data_master[i].results["timeseries"][0,4])[0,:,1] for i in range(self.snapshot_no+1)] 
            Protein_coords_x=[Data_master[i].results["timeseries"][0,7] for i in range(self.snapshot_no+1)]
            Protein_coords_y=[Data_master[i].results["timeseries"][0,7] for i in range(self.snapshot_no+1)]

        pickle_files(f"{self.savingfolder}Voronoi_volumes",Freud_volumes)
        pickle_files(f"{self.savingfolder}Voronoi_cells",Freud_cells)
        pickle_files(f"{self.savingfolder}LPS_phosphate_coords",[LPS_phosphate_coords_x, LPS_phosphate_coords_y])
        pickle_files(f"{self.savingfolder}Protein_BB_coords",Protein_BB_coords)
       
        
        
        return Freud_volumes, Freud_cells, [LPS_phosphate_coords_x, LPS_phosphate_coords_y],[Protein_coords_x,Protein_coords_y],Protein_BB_coords
        
        

                         
#            return box_coords,freud_coords, freud_volumes, freud_cells,LPS_phosphate_coords,POPE_phosphate_coords,POPG_phosphate_coords,PMB1_upper_coords,PMB1_lower_coords
        
                         
       

                         
        
    