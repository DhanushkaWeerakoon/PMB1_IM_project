import MDAnalysis as mda
from MDAnalysis.analysis import density
from lipyphilic.lib.assign_leaflets import AssignLeaflets, AssignCurvedLeaflets
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis import transformations as trans
import numpy as np
import pickle

class TwoD_density():
    
    def __init__(self,tpr,traj,step,xdim,zdim,start=0,stop=-1,verbose=True,sel=None,selname=None,update=False):
        self.tpr=tpr
        self.traj=traj
        self.start=start
        self.stop=stop
        self.step=step
        self.universe=mda.Universe(tpr,traj,continuous=True)
        self.xdim=xdim
        self.ydim=xdim
        self.zdim=zdim
        self.COM=np.array([xdim/2,xdim/2,zdim/2])
        self.verbose=verbose
        self.sel=sel
        self.selname=selname
        self.update=update
        

    def densities(self):
        Selection=self.universe.select_atoms(self.sel,updating=self.update)
        
        #workflow=[trans.unwrap(self.universe.atoms),trans.wrap(self.universe.atoms,compound='fragments')]
        #self.universe.trajectory.add_transformations(*workflow)
        
        dens=density.DensityAnalysis(Selection,delta=1.0,xdim=self.xdim,ydim=self.ydim,zdim=self.zdim,
                                        gridcenter=self.COM)
        dens.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)
        
        grid=dens.results.density.grid        
        avg=grid.mean(axis=-1)
                            
        return avg
        

        











