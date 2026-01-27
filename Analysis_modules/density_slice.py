import numpy as np
import freud
import pickle

import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis import density
import MDAnalysis.transformations as trans

from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.base import (AnalysisBase, AnalysisFromFunction, analysis_class)


from lipyphilic.lib.assign_leaflets import AssignLeaflets, AssignCurvedLeaflets

def pickle_files(filename,variable):
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

class Density_slice:
    def __init__(self,tpr,traj,water_sel,phosphate_sel,boxdim,axis,selname,saving_folder,start,stop,step=1,verbose=True):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj,continuous=True)
        self.water_sel=water_sel
        self.phosphate_sel=phosphate_sel
        self.boxdim=boxdim
        self.axis=axis
        self.verbose=verbose
        self.start=start
        self.stop=stop
        self.step=step
        self.savingfolder=saving_folder
        self.selname=selname
        
    def density_slice_calc(self):
        workflow=[trans.unwrap(self.universe.atoms),trans.wrap(self.universe.atoms,compound='fragments')]
        self.universe.trajectory.add_transformations(*workflow)
        self.universe.trajectory[0]
        
        Water_sel_middle=self.universe.select_atoms(f"{self.water_sel}",updating=True)
        Phosphates=self.universe.select_atoms(f"{self.phosphate_sel}")
        
        Water_coords=Water_sel_middle.positions
        Phosphates_coords=Phosphates.positions
        
        print(self.universe.dimensions)
        dens_water=density.DensityAnalysis(Water_sel_middle,delta=1.0,xdim=self.boxdim[0],ydim=self.boxdim[0],zdim=self.boxdim[2],gridcenter=[self.boxdim[0]/2,self.boxdim[0]/2,self.boxdim[2]/2])
        dens_phosphates=density.DensityAnalysis(Phosphates,delta=1.0,xdim=self.boxdim[0],ydim=self.boxdim[0],zdim=self.boxdim[2],gridcenter=[self.boxdim[0]/2,self.boxdim[0]/2,self.boxdim[2]/2])
        
        dens_water.run(start=self.start,stop=self.stop,verbose=self.verbose)
        dens_phosphates.run(start=self.start,stop=self.stop,verbose=self.verbose)
        
        grid_water=dens_water.results.density.grid
        grid_phosphates=dens_phosphates.results.density.grid
        
        avg_water=grid_water.mean(axis=self.axis)
        avg_phosphates=grid_phosphates.mean(axis=self.axis)
        
        
        avg_water_trans=np.transpose(avg_water)
        avg_phosphates_trans=np.transpose(avg_phosphates)
            
        pickle_files(f"{self.savingfolder}{self.selname}",[avg_water_trans,avg_phosphates_trans])
        pickle_files(f"{self.savingfolder}Coordinates",[Water_coords,Phosphates_coords])
        return avg_water_trans,avg_phosphates_trans
        
    
        

   
    
