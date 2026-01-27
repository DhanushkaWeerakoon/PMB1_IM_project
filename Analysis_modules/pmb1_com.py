import numpy as np
import MDAnalysis as mda
from lipyphilic.transformations import nojump
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
import pickle

# Helper functions
def COM_sel(u, group_a, subdivision,**kwargs):
    x_total=[]
    
    x=group_a.center_of_mass(compound=f"{subdivision}")
    
    if subdivision=="fragments":
        frag_no=group_a.fragments
    else:
        frag_no=group_a.residues
        
    mask=kwargs.get('mask', None)
    x_pre=x[mask]
    x_total.append(x_pre)
    return x_total


def COM_phosphates(u,Upper_phosphates,Lower_phosphates):
    Upper_positions=[]
    Lower_positions=[]
    
    Upper_positions.append(Upper_phosphates.center_of_mass(compound='group'))
    Lower_positions.append(Lower_phosphates.center_of_mass(compound='group'))
    
    return Upper_positions,Lower_positions

Calculate_COMs=analysis_class(COM_sel)
Calculate_COM_phosphates=analysis_class(COM_phosphates)

def pickle_files(filename,variable):
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

class COM():
    def __init__(self,tpr,traj,saving_folder,sel,selname,phosphates="name PO1 PO2 PO4",start=0,stop=-1,step=1,verbose=True,xy=False,z=True,upper=True,phos_dat=False,subdivision='fragments',mask_frame=499):
        self.tpr=tpr
        self.traj=traj
        self.savingfolder=saving_folder
        self.universe=mda.Universe(tpr,traj)
        # Selection of lipid phosphates for identifying upper and lower leaflets using MDA Leaflet Finder
        self.phosphates=phosphates
        # Selection for COM calculation, typically a group of lipids or peptides
        self.sel=sel
        # Whether to calculate COM per residue or per fragment - per fragment by default
        self.subdivision=subdivision
        # Base name for files
        self.selname=selname
        self.verbose=verbose
        # Trajectory controls
        self.start=start
        self.stop=stop
        self.step=step
        # Whether to calculate x, y or z COM coordinates - by default, just z, not xy
        self.xy=xy
        self.z=z
        self.upper=upper
        # Whether to save phosphate COM positions or not
        self.phos_dat=phos_dat
        self.subdivision=subdivision
        self.mask_frame=mask_frame
    
    # Calculates COM for
    # Calculates COM for lipids/peptides - typically grouped by either fragments or residues.
# For working 
    def COM_calculator(self):
        self.universe.trajectory[0]
        L = LeafletFinder(self.universe,self.phosphates,pbc="True")
        leaflet0 = L.groups(0)
        leaflet1 = L.groups(1)
        
        Selection=self.universe.select_atoms(self.sel)
        
        Upper_leaflet_phosphates=leaflet0.atoms
        Lower_leaflet_phosphates=leaflet1.atoms
        
        if self.z==True:
        
            for ts in self.universe.trajectory[self.mask_frame:self.mask_frame+1]:
                COM_PMB1s_ref=Selection.center_of_mass(compound=self.subdivision)
                Upper_leaflet_positions=Upper_leaflet_phosphates.center_of_mass()
                Lower_leaflet_positions=Lower_leaflet_phosphates.center_of_mass()
                Middle_z=(Upper_leaflet_positions[2]+Lower_leaflet_positions[2])/2
                Upper_limit=self.universe.dimensions[2]+5
                print(Upper_limit)                
            if self.upper==True:
                Mask=(COM_PMB1s_ref[:,2]>Middle_z) & (COM_PMB1s_ref[:,2]<Upper_limit) 
                #Mask=(COM_PMB1s_ref[:,2]>0) & (COM_PMB1s_ref[:,2]<150)
            elif self.upper==False:
                Mask=(COM_PMB1s_ref[:,2]<Middle_z) | (COM_PMB1s_ref[:,2]>Upper_limit)
                #Mask=(COM_PMB1s_ref[:,2] < 0) | (COM_PMB1s_ref[:,2] > 150)
            
            print(Mask)
            Selection_COMs=Calculate_COMs(u=self.universe,group_a=Selection,subdivision=self.subdivision,mask=Mask)
            Selection_COMs.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)
        
            print(np.shape(Selection_COMs.results["timeseries"]))
            PMB1_no=sum((x==True) for x in Mask)
        
            if self.xy==True:
                Total_selection=[Selection_COMs.results["timeseries"][:,0,x,:] for x in range(PMB1_no)]
        
            elif self.xy==False:
                Total_selection=[Selection_COMs.results["timeseries"][:,0,x,2] for x in range(PMB1_no)]
        
        
        else:
            Selection_COMs=Calculate_COMs(u=self.universe,group_a=Selection,subdivision=self.subdivision)
            Selection_COMs.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)
            if self.subdivision=="fragments":
                frag_no=len(Selection.fragments)
            else:
                frag_no=len(Selection.residues)
            print(frag_no)
            Total_selection = [Selection_COMs.results["timeseries"][:,0,0,x,:] for x in range(frag_no)]
            
            
            print(np.shape(Selection_COMs.results["timeseries"]))
        
        pickle_files(f"{self.savingfolder}{self.selname}",Total_selection)
        Time=Selection_COMs.results["times"]
        
        if self.phos_dat==True:
            Phosphate_calculations=Calculate_COM_phosphates(self.universe,Upper_leaflet_phosphates,Lower_leaflet_phosphates)
            Phosphate_calculations.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)
            Phosphate_selection=Phosphate_calculations.results["timeseries"]
            
            pickle_files(f"{self.savingfolder}Phosphates",Phosphate_selection)
            pickle_files(f"{self.savingfolder}Times",Time)
        
            return Time,Phosphate_selection, Total_selection
        
        elif self.phos_dat==False:
            return Time,Total_selection
