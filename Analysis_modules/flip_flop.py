import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
import pickle

def pickle_files(filename,variable):
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

def Flip_flop_calc(universe,phosphates):
    L=LeafletFinder(universe,phosphates,pbc=True)
    Leaflet0=L.groups(0)
    Leaflet1=L.groups(1)
    
    Leaflet0_POPE_num=len(Leaflet0.select_atoms("resname POPE",updating=True).residues)
    Leaflet0_POPG_num=len(Leaflet0.select_atoms("resname POPG",updating=True).residues)

    Leaflet1_POPE_num=len(Leaflet1.select_atoms("resname POPE",updating=True).residues)
    Leaflet1_POPG_num=len(Leaflet1.select_atoms("resname POPG",updating=True).residues)
    
    print(Leaflet0_POPE_num)
    
    return Leaflet0_POPE_num,Leaflet1_POPE_num,Leaflet0_POPG_num,Leaflet1_POPG_num

class Flip_flop:
    def __init__(self,tpr,traj,phosphate_sel,midpoint,upper,selname,saving_folder,start,stop,step=1,verbose=True):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj,continuous=True)
        self.phosphate_sel=phosphate_sel
        self.midpoint=midpoint
        self.upper=upper
        self.verbose=verbose
        self.start=start
        self.stop=stop
        self.step=step
        self.savingfolder=saving_folder
        self.selname=selname
    

    def flip_flop_calc(self):
        self.universe.trajectory[0]
        Phosphate_upper_bilayer=self.universe.select_atoms(f"({self.phosphate_sel}) and (prop z > {self.midpoint})")
        Phosphate_lower_bilayer=self.universe.select_atoms(f"({self.phosphate_sel}) and (prop z < {self.midpoint})")
        
        
        Calculate_Flips=analysis_class(Flip_flop_calc)
        
        
        if self.upper==True:
            Flip_flops=Calculate_Flips(universe=self.universe,phosphates=Phosphate_upper_bilayer)
        
        if self.upper==False:
            Flip_flops=Calculate_Flips(universe=self.universe,phosphates=Phosphate_lower_bilayer)
        Flip_flops_results=Flip_flops.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)
           
        
        pickle_files(f"{self.savingfolder}{self.selname}",[Flip_flops_results.results])
        return Flip_flops_results