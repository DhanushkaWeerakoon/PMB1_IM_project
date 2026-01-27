import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
import pickle

# Helper functions
def COM_sel(u, group_a, subdivision):
    x_total=[]
    
    x=group_a.center_of_mass(compound=f"{subdivision}")
    
    if subdivision=="fragments":
        frag_no=group_a.fragments
    else:
        frag_no=group_a.residues
        
    x_total.append(x)
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
    def __init__(self,tpr,traj,sel,phosphates="name PO1 PO2 PO4",start=0,stop=-1,step=1,subdivision='fragments',verbose=True):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj)
        self.phosphates=phosphates
        self.sel=sel
        self.subdivision=subdivision
        self.verbose=verbose
        self.start=start
        self.stop=stop
        self.step=step
    
    # Calculates COM for
    # Calculates COM for lipids/peptides - typically grouped by either fragments or residues.
# For working 
    def COM_calculator(self):
        self.universe.trajectory[0]
        
        Selection=self.universe.select_atoms(self.sel)
        Selection_COMs=Calculate_COMs(u=self.universe,group_a=Selection,subdivision=self.subdivision)
        Selection_COMs.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)
        
        print(np.shape(Selection_COMs.results["timeseries"]))
        
        if self.subdivision=="fragments":
            PMB1_no=len(Selection.fragments)
        else:
            PMB1_no=len(Selection.residues)
                
        Total_selection=[Selection_COMs.results["timeseries"][:,0,x,:] for x in range(PMB1_no)]
        Time=Selection_COMs.results["times"]
        
        
        L = LeafletFinder(self.universe,self.phosphates,pbc="True")
        leaflet0 = L.groups(0)
        leaflet1 = L.groups(1)
            
        Upper_leaflet_phosphates=leaflet0.atoms
        Lower_leaflet_phosphates=leaflet1.atoms
            
        Phosphate_calculations=Calculate_COM_phosphates(self.universe,Upper_leaflet_phosphates,Lower_leaflet_phosphates)
        Phosphate_calculations.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)
        Phosphate_selection=Phosphate_calculations.results["timeseries"]
                    
        return Time,Phosphate_selection, Total_selection
        

