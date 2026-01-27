import MDAnalysis as mda
#from MDAnalysis.tests.datafiles import TPR, XTC
from MDAnalysis.analysis import rdf
import numpy as np

from MDAnalysis.analysis.base import (AnalysisBase,
                                      AnalysisFromFunction,
                                      analysis_class)

class Sugar_ROG():
    
    def __init__(self,tpr,traj,step,saving_folder,sugar_sel="resname RAMP and name S*",verbose=True):
        self.tpr=tpr
        self.traj=traj
        self.sugar_sel=sugar_sel
        self.step=step
        self.verbose=verbose
        self.universe=mda.Universe(tpr,traj)
        self.savingfolder=saving_folder
        
    def rog_class(self): 
        Sugar=self.universe.select_atoms(self.sugar_sel)
        def rog():
            ROG=Sugar.radius_of_gyration(compound='residues')
            return ROG
        ROG=analysis_class(rog)
        results=ROG(self.universe.trajectory)
        results.run(step=self.step,verbose=self.verbose)
        
        Times=results.results["times"]
        Sugar_Rgyr=results.results["timeseries"]
        
        np.savetxt(self.savingfolder+"Sugar_ROG.dat", np.c_[Times,Sugar_Rgyr])
        return Times, Sugar_Rgyr
    
    