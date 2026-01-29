"""
This module provides tools for calculating 2D density slices of water and
phosphates in membrane systems. Creates density grids and averages along
a specified axis to produce 2D projections, useful for visualizing water
penetration into membranes and phosphate distributions.
"""

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
     """
    Save a Python variable to a pickle file.
    
    Parameters
    ----------
    filename : str
        Path to output pickle file.
    variable : any
        Python object to serialize and save.
        
    Notes
    -----
    Writes file in binary mode ('wb').
    """
    
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

class Density_slice:
     """    
    This class computes 3D density grids for water and phosphate selections,
    then averages along a specified axis to create 2D density projections.
    Particularly useful for visualizing water penetration into membranes,
    pore formation, and phosphate distributions in membrane simulations.

    Adapted from density calculation code in MDAnalysis tutorials.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    water_sel : str
        MDAnalysis selection string for water molecules/beads.
    phosphate_sel : str
        MDAnalysis selection string for phosphate atoms/beads.
    boxdim : array-like of float
        Simulation box dimensions [x, y, z] in Angstroms.
        Assumes square xy-plane (x == y).
    axis : int
        Axis along which to average the 3D density grid:
        - 0: average along x-axis (yz-plane projection)
        - 1: average along y-axis (xz-plane projection)
        - 2: average along z-axis (xy-plane projection)
    selname : str
        Base name for output pickle file.
    saving_folder : str
        Directory path where density results will be saved.
    start : int
        First frame to analyze.
    stop : int
        Last frame to analyze.
    step : int, default=1
        Analyze every nth frame.
    verbose : bool, default=True
        Print progress information during analysis.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    savingfolder : str
        Directory for saving results.
    selname : str
        Base filename for outputs.
    
    Notes
    -----
    Coordinate Transformations:
    - unwrap: Remove periodic boundary artifacts
    - wrap (compound='fragments'): Recenter fragments in box
    - These prevent molecules from being split across boundaries
    
    Grid Properties:
    - Grid spacing (delta): 1.0 Å in all dimensions
    - Grid dimensions: boxdim[0] × boxdim[0] × boxdim[2]
    - Grid centered at: [x/2, y/2, z/2]
    - Assumes square xy-plane (x == y)
    
    Density Calculation:
    - 3D histogram of atom positions over trajectory
    - Normalized by grid volume and number of frames
    - Units: typically number/Å³ (depends on selection)
    
    Averaging:
    - Averages 3D density grid along specified axis
    - Produces 2D projection (slice) in remaining plane
    
    Output Files:
    Two pickle files are created:
    1. {saving_folder}{selname}: Contains [water_density_2D, phosphate_density_2D]
    2. {saving_folder}Coordinates: Contains [water_coords, phosphate_coords]
       from first frame (for reference)
    
    Updating Selections:
    - Water selection uses updating=True to handle dynamic groups
    - Phosphate selection is static (updating=False)
    - Important for systems where water molecules cross boundaries
    """
    
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
        """
        Calculate 2D density slices for water and phosphates.
        
        Computes 3D density grids for water and phosphate selections over
        the specified trajectory frames, then averages along the specified
        axis to create 2D density projections. Also saves initial coordinates
        for reference.
        
        Returns
        -------
        avg_water_trans : ndarray
            Transposed 2D water density array after averaging along axis.
            Shape depends on axis parameter:
            - axis=0: (z, y)
            - axis=1: (z, x)
            - axis=2: (y, x)
        avg_phosphates_trans : ndarray
            Transposed 2D phosphate density array after averaging along axis.
            Same shape as avg_water_trans.
     
        Notes
        -----
        Workflow:
        1. Apply coordinate transformations (unwrap then wrap)
        2. Move to first frame in trajectory
        3. Select water molecules (with updating) and phosphates (static)
        4. Store initial coordinates for reference
        5. Create DensityAnalysis objects for both selections
        6. Run density calculations over specified frame range
        7. Extract 3D density grids
        8. Average along specified axis to get 2D projections
        9. Transpose results for convenient plotting
        10. Save results to pickle files
        
        Coordinate Transformations:
        
        
        Grid Parameters:
        - delta=1.0: 1 Å spacing in all dimensions
        - xdim=ydim=boxdim[0]: Square grid in xy-plane
        - zdim=boxdim[2]: Z-dimension from input
        - gridcenter: Center of box [x/2, y/2, z/2]
        
        Averaging:
        The 3D grid is averaged along self.axis:
        - axis=0: avg(x) → yz-plane (side view)
        - axis=1: avg(y) → xz-plane (side view)
        - axis=2: avg(z) → xy-plane (top view)
        
        Transposition:
        Results are transposed to have shape (dim2, dim1) for plotting
        convenience with matplotlib's imshow (origin='lower').

        
        
        
        """

        # Apply coordinate transformations
        workflow=[trans.unwrap(self.universe.atoms),trans.wrap(self.universe.atoms,compound='fragments')]
        self.universe.trajectory.add_transformations(*workflow)
        self.universe.trajectory[0]

        # Select atoms (water with updating, phosphates static)
        
        Water_sel_middle=self.universe.select_atoms(f"{self.water_sel}",updating=True)
        Phosphates=self.universe.select_atoms(f"{self.phosphate_sel}")

        # Store initial coordinates for reference
        Water_coords=Water_sel_middle.positions
        Phosphates_coords=Phosphates.positions
        
        # Create density analysis objects
        dens_water=density.DensityAnalysis(Water_sel_middle,delta=1.0,xdim=self.boxdim[0],ydim=self.boxdim[0],zdim=self.boxdim[2],gridcenter=[self.boxdim[0]/2,self.boxdim[0]/2,self.boxdim[2]/2])
        dens_phosphates=density.DensityAnalysis(Phosphates,delta=1.0,xdim=self.boxdim[0],ydim=self.boxdim[0],zdim=self.boxdim[2],gridcenter=[self.boxdim[0]/2,self.boxdim[0]/2,self.boxdim[2]/2])

        # Run density calculations
        dens_water.run(start=self.start,stop=self.stop,verbose=self.verbose)
        dens_phosphates.run(start=self.start,stop=self.stop,verbose=self.verbose)

        # Extract 3D density grids
        grid_water=dens_water.results.density.grid
        grid_phosphates=dens_phosphates.results.density.grid

        # Average along specified axis to get 2D slices
        avg_water=grid_water.mean(axis=self.axis)
        avg_phosphates=grid_phosphates.mean(axis=self.axis)
        
        # Transpose for convenient plotting
        avg_water_trans=np.transpose(avg_water)
        avg_phosphates_trans=np.transpose(avg_phosphates)

        # Save results to pickle files
        pickle_files(f"{self.savingfolder}{self.selname}",[avg_water_trans,avg_phosphates_trans])
        pickle_files(f"{self.savingfolder}Coordinates",[Water_coords,Phosphates_coords])
        return avg_water_trans,avg_phosphates_trans
        
    
        

   
    
