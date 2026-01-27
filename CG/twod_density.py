"""
Module calculates 2D density maps of atomic selections in membrane simulations using MDAnalysis, with support for leaflet assignment and coordinate transformations.
"""

import MDAnalysis as mda
from MDAnalysis.analysis import density
from lipyphilic.lib.assign_leaflets import AssignLeaflets, AssignCurvedLeaflets
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis import transformations as trans
import numpy as np
import pickle

class TwoD_density():
    """
    Calculate two-dimensional density maps of atomic selections by averaging a 3D density
    grid along the z-axis.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    step : int
        Analyze every nth frame.
    xdim : float
        Size of the grid in x (and y) dimensions (Å). The grid is assumed
        to be square in the xy-plane.
    zdim : float
        Size of the grid in z dimension (Å).
    start : int, default=0
        First frame to analyze.
    stop : int, default=-1
        Last frame to analyze.
    verbose : bool, default=True
        Print progress information during analysis.
    sel : str, optional
        MDAnalysis selection string for atoms to include in density calculation.
    update : bool, default=False
        Whether to update the selection each frame. Set to True for
        dynamic selections.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory with continuous unwrapping enabled.
    xdim : float
        Grid size in x direction.
    ydim : float
        Grid size in y direction (same as xdim for square grid).
    zdim : float
        Grid size in z direction.
    COM : ndarray
        Center of mass coordinates for the grid [xdim/2, xdim/2, zdim/2].
        
    
    Notes
    -----
    - Grid spacing (delta) is fixed at 1.0 Å
    - Density units depend on the selection (typically number/Å³)
    """
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
        """        
        Creates a 3D density grid with 1.0 Å spacing, then averages along
        the z-axis to produce a 2D density map in the xy-plane. This is
        useful for visualizing lateral distributions in membrane systems.
        
        Returns
        -------
        avg : ndarray
            2D array of shape (xdim, ydim) containing the averaged density
            values. Each element represents the mean density at that xy
            position averaged over all z slices.
            
        Notes
        -----
        - Grid spacing (delta) is set to 1.0 Å in all dimensions
        - Grid is centered at [xdim/2, ydim/2, zdim/2]
        
        """
        # Select atoms for density calculation
        Selection=self.universe.select_atoms(self.sel,updating=self.update)

        # Calculate 3D density grid
        dens=density.DensityAnalysis(Selection,delta=1.0,xdim=self.xdim,ydim=self.ydim,zdim=self.zdim,
                                        gridcenter=self.COM)
        dens.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)
        
        grid=dens.results.density.grid   

        # Average along z axis to get 2D projection
        avg=grid.mean(axis=-1)
                            
        return avg
        

        











