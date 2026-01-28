"""
Module calculates lipid tail order parameters (SCC), creating 2D spatial maps of order parameters, and tracking molecular positions in membrane systems.

This is an adapted version of the Lipyphilic order parameter module (https://pubs.acs.org/doi/10.1021/acs.jctc.1c00447) enabling 2D order parameters to be calculated for hexa-tail lipids such as RaLPS.

"""

import MDAnalysis as mda
import numpy as np
import scipy

from lipyphilic.lib.order_parameter import SCC
from MDAnalysis.analysis.leaflet import LeafletFinder
from lipyphilic.lib.assign_leaflets import AssignLeaflets, AssignCurvedLeaflets

import pickle

from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)

def weighted_average_6(sn1_scc,sn2_scc,sn3_scc,sn4_scc,sn5_scc,sn6_scc):
    """    
    Combines SCC (carbon-carbon order parameter) data from six lipid tails
    (e.g., for RaLPS lipids) into a single weight-averaged SCC object. Weighting
    is based on the number of tail atoms minus one for each tail.
    
    Based on preexisting Lipyphilic code, adapted for 6-tailed lipids.
    
    Parameters
    ----------
    sn1_scc : lipyphilic.lib.order_parameter.SCC
        SCC object for tail 1 (sn-1 position).
    sn2_scc : lipyphilic.lib.order_parameter.SCC
        SCC object for tail 2 (sn-2 position).
    sn3_scc : lipyphilic.lib.order_parameter.SCC
        SCC object for tail 3 (sn-3 position).
    sn4_scc : lipyphilic.lib.order_parameter.SCC
        SCC object for tail 4 (sn-4 position).
    sn5_scc : lipyphilic.lib.order_parameter.SCC
        SCC object for tail 5 (sn-5 position).
    sn6_scc : lipyphilic.lib.order_parameter.SCC
        SCC object for tail 6 (sn-6 position).
        
    Returns
    -------
    new_scc : lipyphilic.lib.order_parameter.SCC
        New SCC object containing weighted average order parameters.
        Has the same number of frames as input objects and contains
        combined data for all tails.
        
    Notes
    -----
    Workflow:
    1. Extract residue indices from all six tail objects
    2. Create empty SCC array of size (n_residues, n_frames)
    3. For each lipid species:
       a) Determine tail atoms matching the species in qustion for each tail,
       b) Count number of tail atoms per lipid,
       c) Extract SCC values for that species from each tail,
    4. Average snx_species_scc objects together,
    5. Combine atom indices,
    5. Create new SCC object same number of frames as sn1_scc and averaged scc values,
    6. Copy trajectory metadata (start, stop, step, times) from sn1_scc
    7. Return new SCC object.
    
    The weighting by (n_atoms - 1) accounts for the fact that SCC is
    calculated between pairs of carbon atoms, so a tail with N atoms
    has N-1 SCC values.
    
    """

    # Extract residue indices from each tail
    sn1_resindices=sn1_scc.tails.residues.resindices
    sn2_resindices=sn2_scc.tails.residues.resindices
    sn3_resindices=sn3_scc.tails.residues.resindices
    sn4_resindices=sn4_scc.tails.residues.resindices
    sn5_resindices=sn5_scc.tails.residues.resindices
    sn6_resindices=sn6_scc.tails.residues.resindices

    # Combine all residue indices
    combined_resindices=np.unique(np.hstack([sn1_resindices, sn2_resindices,sn3_resindices,sn4_resindices,sn5_resindices,sn6_resindices]))
    n_residues=combined_resindices.size
    scc = np.zeros((n_residues, sn1_scc.n_frames))

    # Process each lipid species separately
    for species in np.unique(np.hstack([sn1_scc.tails.resnames, sn2_scc.tails.resnames,sn3_scc.tails.resnames,sn4_scc.tails.resnames,sn5_scc.tails.resnames,sn6_scc.tails.resnames])):

        # Extract SCC values for species from each tail
        sn1_species_scc=sn1_scc.SCC[sn1_scc.tail_residue_mask[species]]
        sn2_species_scc=sn2_scc.SCC[sn2_scc.tail_residue_mask[species]]
        sn3_species_scc=sn3_scc.SCC[sn3_scc.tail_residue_mask[species]]
        sn4_species_scc=sn4_scc.SCC[sn4_scc.tail_residue_mask[species]]
        sn5_species_scc=sn5_scc.SCC[sn5_scc.tail_residue_mask[species]]
        sn6_species_scc=sn6_scc.SCC[sn6_scc.tail_residue_mask[species]]

        # Calculate number of atoms per lipid for weighting
        sn1_n_atoms_per_lipid=sn1_scc.tails[sn1_scc.tail_atom_mask[species]].n_atoms/len(sn1_species_scc)
        sn2_n_atoms_per_lipid=sn2_scc.tails[sn2_scc.tail_atom_mask[species]].n_atoms/len(sn2_species_scc)
        sn3_n_atoms_per_lipid=sn3_scc.tails[sn3_scc.tail_atom_mask[species]].n_atoms/len(sn3_species_scc)
        sn4_n_atoms_per_lipid=sn4_scc.tails[sn4_scc.tail_atom_mask[species]].n_atoms/len(sn4_species_scc)
        sn5_n_atoms_per_lipid=sn5_scc.tails[sn5_scc.tail_atom_mask[species]].n_atoms/len(sn5_species_scc)
        sn6_n_atoms_per_lipid=sn6_scc.tails[sn6_scc.tail_atom_mask[species]].n_atoms/len(sn6_species_scc)

        # Calculated weighted average (weight = no_atoms - 1)
        species_scc=np.average(np.array([sn1_species_scc, sn2_species_scc,sn3_species_scc,sn4_species_scc,sn5_species_scc,sn6_species_scc]),axis=0,weights=[sn1_n_atoms_per_lipid - 1,sn2_n_atoms_per_lipid - 1,sn3_n_atoms_per_lipid - 1,sn4_n_atoms_per_lipid - 1,sn5_n_atoms_per_lipid - 1,sn6_n_atoms_per_lipid - 1])
        
        # Assign species_scc to the relevant part of the empty SCC array
species_resindices=np.in1d(combined_resindices,sn1_resindices[sn1_scc.tail_residue_mask[species]])
        scc[species_resindices]=species_scc

    sn1_atom_indices = sn1_scc.tails.indices
    sn2_atom_indices = sn2_scc.tails.indices
    sn3_atom_indices = sn3_scc.tails.indices
    sn4_atom_indices = sn4_scc.tails.indices
    sn5_atom_indices = sn5_scc.tails.indices
    sn6_atom_indices = sn6_scc.tails.indices

    # Combine atom indices from all tails
    combined_atom_indices = np.unique(np.hstack([sn1_atom_indices, sn2_atom_indices,sn3_atom_indices,sn4_atom_indices,sn5_atom_indices,sn6_atom_indices]))

    # Create new SCC object containing combined data
    new_scc = SCC(universe=sn1_scc.u,tail_sel=f"index {' '.join(combined_atom_indices.astype(str))}",)
    new_scc.start, new_scc.stop, new_scc.step = sn1_scc.start, sn1_scc.stop, sn1_scc.step
    new_scc.frames = np.arange(new_scc.start, new_scc.stop, new_scc.step)
    new_scc.n_frames = new_scc.frames.size
    new_scc.times = sn1_scc.times
    new_scc._trajectory = sn1_scc._trajectory
    new_scc.SCC = scc
     
    return new_scc
        

def Custom_binning_function(scc_object1,scc_object2,lipid_sel,startframe,endframe,stepframe,bins,statistic,filterby):
    """
    Bins lipid molecules by the xy-coordinates into bins in 2D space and calculates mean/
    median/standard deviation of lipid order parameters within each bi. Handles RaLPS and \
    phospholipids separately, with optional filtering for phospholipids. Uses interpolation 
    to fill NaN values in sparse regions.

    Adapted from pre-existing Lipyphilic code.
    
    Parameters
    ----------
    scc_object1 : lipyphilic.lib.order_parameter.SCC
        SCC object for first lipid population (e.g., RaLPS molecules).
    scc_object2 : lipyphilic.lib.order_parameter.SCC
        SCC object for second lipid population (e.g., phospholipids).
        Must use same timeframe as scc_object1.
    lipid_sel : str
        MDAnalysis selection string for lipid headgroup atoms.
    startframe : int
        First frame to include in analysis.
    endframe : int
        Last frame to include in analysis.
    stepframe : int
        Use every nth frame.
    bins : int or array-like
        Number of bins or bin edges for 2D histogram.
    statistic : str or callable
        Statistic to compute in each bin (e.g., 'mean', 'median', 'std').
    filterby : array-like
        Boolean mask for filtering scc_object2 lipids. Should have shape
        (n_lipids, n_frames). Typically used to select upper leaflet lipids.
        
    Returns
    -------
    stat : ndarray
        2D array of bins containing relevant statistic after direct interpolation.
        Shape: (bins, bins).
    stat_tiled : ndarray
        2D array of binned statistics after tiled interpolation (accounts
        for periodic boundary conditions). 
        Shape: (bins, bins).
    x_edges : ndarray
        Bin edges along x-axis.
    y_edges : ndarray
        Bin edges along y-axis.
        
    Notes
    -----
    Workflow:
    1. Filter lipids by selection and leaflet membership
    2. Extract frame indices for specified time range
    3. Use middle frame for spatial coordinates (assumes system is equilibrated and
    lipids do not diffuse significantly during analysis period)
    4. Calculate center of mass for each lipid residue (unwrap = False)
    5. Apply periodic boundary corrections to coordinates
    6. Calculate time-averaged SCC for each lipid
    7. Combine filtered populations and bin by xy-coordinates
    8. Interpolate NaN values using tiled grid (for PBC) or direct grid
    """
    

    # Select lipid populations
    filterby=np.array(filterby)
    PLs = scc_object2.tails.residues.atoms.select_atoms(lipid_sel) 
    RAMP= scc_object1.tails.residues.atoms.select_atoms(lipid_sel)

    # Filter residues based on selections
    keep_RAMP = np.in1d(scc_object1.tails.residues.resindices, RAMP.residues.resindices)
    keep_PLs = np.in1d(scc_object2.tails.residues.resindices, PLs.residues.resindices)

    # Handle frame indexing
    start, stop, step = scc_object1.u.trajectory.check_slice_indices(startframe, endframe, stepframe)
    frames = np.arange(start, stop, step)
    keep_frames = np.in1d(scc_object1.frames, frames)
    frames = scc_object1.frames[keep_frames]

    # Extract SCC data for selected frames
    scc_RAMP= scc_object1.SCC[keep_RAMP][:, keep_frames]
    scc_PLs= scc_object2.SCC[keep_PLs][:, keep_frames]

    # Use middle frame as reference for coordinates   
    mid_frame = frames[frames.size // 2]
    mid_frame_index = np.min(np.where(scc_object1.frames == mid_frame))
    filterby = filterby[keep_PLs][:, mid_frame_index]
    scc_object1.u.trajectory[mid_frame]
    scc_object2.u.trajectory[mid_frame]

    # Group atoms by residues
    residues_PLs = PLs.groupby("resindices")
    residues_RAMP=RAMP.groupby("resindices")

    # Calculate centre of mass for each residue
    PLs_com = np.array([residues_PLs[res].center_of_mass(unwrap=False) for res in residues_PLs])
    RAMP_com = np.array([residues_RAMP[res].center_of_mass(unwrap=False) for res in residues_RAMP])

    # Apply periodic boundary conditions
    for dim in range(3):
        PLs_com[:, dim][PLs_com[:, dim] > scc_object2.u.dimensions[dim]] -= scc_object2.u.dimensions[dim]
        PLs_com[:, dim][PLs_com[:, dim] < 0.0] += scc_object2.u.dimensions[dim]
        
        RAMP_com[:, dim][RAMP_com[:, dim] > scc_object1.u.dimensions[dim]] -= scc_object1.u.dimensions[dim]
        RAMP_com[:, dim][RAMP_com[:, dim] < 0.0] += scc_object1.u.dimensions[dim]

    # Extract x, y x coordinates
    PLs_xpos, PLs_ypos, _ = PLs_com.T 
    RAMP_xpos, RAMP_ypos, _ = RAMP_com.T 

    # Calculate time-averaged SCC values
    values_PLs = np.mean(scc_PLs, axis=1)
    values_RAMP = np.mean(scc_RAMP,axis=1)
    
    # Combine filtered lipids
    lipids_xpos=np.hstack([PLs_xpos[filterby],RAMP_xpos])
    lipids_ypos=np.hstack([PLs_ypos[filterby],RAMP_ypos])
    values=np.hstack([values_PLs[filterby],values_RAMP])

    # Calculate 2D binned statistics
    stat, x_edges, y_edges, _ = scipy.stats.binned_statistic_2d(
            x=lipids_xpos,
            y=lipids_ypos,
            values=values,
            bins=bins,statistic=statistic
        )
    
    statistic_nbins_x, statistic_nbins_y = stat.shape

    # Interpolate NaN using tiled grid
    stat_tiled = np.tile(stat,reps=(3,3)) 
    x,y=np.indices(stat_tiled.shape) 
    stat_tiled[np.isnan(stat_tiled)]=scipy.interpolate.griddata(
    (x[~np.isnan(stat_tiled)], y[~np.isnan(stat_tiled)]),  # points we know
           stat_tiled[~np.isnan(stat_tiled)],                     # values we know
        (x[np.isnan(stat_tiled)], y[np.isnan(stat_tiled)]),    # points to interpolate
           method="linear", # other options available, see interpolate fxn
       )

    # Extract central tile after interpolation
    stat_tiled = stat_tiled[statistic_nbins_x:statistic_nbins_x * 2, statistic_nbins_y:statistic_nbins_y * 2]    

    #Interpolate NaN values in original grid
    x,y=np.indices(stat.shape) 
    stat[np.isnan(stat)]=scipy.interpolate.griddata(
       (x[~np.isnan(stat)], y[~np.isnan(stat)]),  # points we know
       stat[~np.isnan(stat)],                     # values we know
        (x[np.isnan(stat)], y[np.isnan(stat)]),    # points to interpolate
        method="linear", # other options available, see interpolate fxn
       )
    
    stat = stat[statistic_nbins_x:statistic_nbins_x * 2, statistic_nbins_y:statistic_nbins_y * 2]    
    
    return stat, stat_tiled,x_edges, y_edges


class Positions(AnalysisBase):
    """    
    This class extracts and stores coordinates of various moieties over the course of a
    simulation. Uses MDAnalysis LeafletFinder to identify which leaflet lipids belong to.

    Inherits from MDAnalysis AnalysisBase class.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    phosphates : str, default="name PO1 PO2 PO4"
        MDAnalysis selection string for phosphate atoms used in
        leaflet identification.
    Protein_sel : str, optional
        MDAnalysis selection string for protein atoms to track.
    Upper_PMB1s_sel : str, optional
        MDAnalysis selection string for upper leaflet-bound PMB1 peptides.
    Lower_PMB1s_sel : str, optional
        MDAnalysis selection string for lower leaflet-bound PMB1 peptides.
    **kwargs : dict
        Additional keyword arguments passed to AnalysisBase.__init__,
        such as start, stop, step, verbose.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    LPS_phosphate_coords : list of ndarray
        List of coordinate arrays for LPS phosphates (PO1, PO2) in
        upper leaflet at each frame.
    POPE_phosphate_coords : list of ndarray
        List of coordinate arrays for POPE phosphates (PO4) in
        upper leaflet at each frame.
    POPG_phosphate_coords : list of ndarray
        List of coordinate arrays for POPG phosphates (PO4) in
        upper leaflet at each frame.
    PMB1_upper_coords : list of ndarray
        List of coordinate arrays for upper PMB1 peptides at each frame
        (if Upper_PMB1s_sel provided).
    PMB1_lower_coords : list of ndarray
        List of coordinate arrays for lower PMB1 peptides at each frame
        (if Lower_PMB1s_sel provided).
    Protein_coords : list of ndarray
        List of coordinate arrays for protein at each frame
        (if Protein_sel provided).
        
    Notes
    -----
    - Only tracks upper leaflet lipids.
    - Coordinate arrays have shape (n_atoms, 3) for each frame
    - PMB1 peptides can be tracked in both leaflets separately
    """
    def __init__(self,tpr,traj,phosphates="name PO1 PO2 PO4",Protein_sel=None,Upper_PMB1s_sel=None,Lower_PMB1s_sel=None,**kwargs):
        universe=mda.Universe(tpr,traj)
        super(Positions,self).__init__(universe.trajectory,**kwargs)
        self.universe=universe
        self.phosphates=phosphates
        self.Upper_PMB1s_sel=Upper_PMB1s_sel
        self.Lower_PMB1s_sel=Lower_PMB1s_sel
        self.Protein_sel=Protein_sel
        
    def _prepare(self):
        """
        Initialize coordinate storage lists.
        
        Called automatically by AnalysisBase.run() before analysis begins.
        """
        self.LPS_phosphate_coords=[]
        self.POPE_phosphate_coords=[]
        self.POPG_phosphate_coords=[]
        self.PMB1_upper_coords=[]
        self.PMB1_lower_coords=[]
        self.Protein_coords=[]

    def _single_frame(self):
        """        
        Identifies upper leaflet using LeafletFinder, then extracts
        coordinates for upper leaflet LPS, POPE, and POPG phosphates. 
        Also extracts PMB1 and protein coordinates if selections are provided.
        
        Called automatically by AnalysisBase.run() for each frame.
        
        Notes
        -----
        - LPS phosphates: PO1 and PO2 atoms
        - POPE/POPG phosphates: PO4 atoms
        - Coordinates are appended to respective lists
        """

        # Identify leaflets
        L=LeafletFinder(self.universe,self.phosphates,pbc=True)
        self.Upper_leaflet=L.groups(0)

        # Extract upper leaflet phosphate coordinates
        self.LPS_phosphates_upper=self.Upper_leaflet.select_atoms("name PO1 PO2")
        self.POPE_phosphates_upper=self.Upper_leaflet.select_atoms('resname POPE and name PO4')
        self.POPG_phosphates_upper=self.Upper_leaflet.select_atoms('resname POPG and name PO4')
        self.LPS_phosphate_coords.append(self.LPS_phosphates_upper.positions)
        self.POPE_phosphate_coords.append(self.POPE_phosphates_upper.positions)
        self.POPG_phosphate_coords.append(self.POPG_phosphates_upper.positions)

        # Extract polymyxin B1 coordinates if selections provided
        if self.Upper_PMB1s_sel !=None:
            self.Upper_PMB1s=self.universe.select_atoms(self.Upper_PMB1s_sel)
            self.PMB1_upper_coords.append(self.Upper_PMB1s.positions)
    
        if self.Lower_PMB1s_sel != None:
            self.Lower_PMB1s=self.universe.select_atoms(self.Lower_PMB1s_sel)
            self.PMB1_lower_coords.append(self.Lower_PMB1s.positions)

        # Extract protein coordinates if selection provided
        if self.Protein_sel != None:
            self.Protein=self.universe.select_atoms(self.Protein_sel)
            self.Protein_coords.append(self.Protein.positions)


class Order_parameter():
    """
    This class computes carbon-carbon order parameters (SCC) for lipid
    tails, handles complex lipids with multiple tails (e.g., RaLPS with
    6 tails), assigns lipids to leaflets, and creates 2D spatial maps
    of order parameters using binned statistics.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    phosphates : str, default="name PO1 PO2 PO4"
        MDAnalysis selection string for phosphate atoms used in
        leaflet assignment.
    PL_sel : str, default="resname POPE POPG"
        Selection for phospholipids (2-tailed lipids).
    RAMP_sel : str, default="resname RAMP"
        Selection for RaLPS lipids (6-tailed lipids).
    lipid_sel : str, default="resname RAMP POPE POPG"
        Selection for all lipids to include in spatial analysis.
    start : int, default=0
        First frame to analyze.
    stop : int, default=-1
        Last frame to analyze (-1 for all frames).
    step : int, default=1
        Analyze every nth frame.
    verbose : bool, default=True
        Print progress information during analysis.
    no_bins : int, default=200
        Number of bins for 2D spatial mapping.
    stat : str, default="mean"
        Statistic to compute in each bin ('mean', 'median', 'std', etc.).
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    leaflets : lipyphilic.lib.assign_leaflets.AssignLeaflets
        Leaflet assignment object.
    scc_PL_ave : lipyphilic.lib.order_parameter.SCC
        Averaged SCC object for phospholipid tails (sn-1 and sn-2).
    scc_RAMP_ave : lipyphilic.lib.order_parameter.SCC
        Averaged SCC object for RaLPS tails (all 6 tails).
    mean : ndarray
        2D array of spatially binned order parameter values.
    mean_tiled : ndarray
        2D array after tiled interpolation (accounts for PBC).
    x_edges : ndarray
        Bin edges along x-axis.
    y_edges : ndarray
        Bin edges along y-axis.
            
    Notes
    -----
    - Only upper leaflet lipids are included in spatial mapping
    - Tail selections use wildcard notation: ??A selects all atoms
      with names ending in 'A' (e.g., C1A, C2A, C3A)
    - SCC values range from -0.5 (perpendicular to bilayer normal) to
      1.0 (parallel to bilayer normal)
    - Higher SCC values indicate more ordered tails
    """
    
    def __init__(self,tpr,traj,phosphates="name PO1 PO2 PO4",PL_sel="resname POPE POPG",RAMP_sel="resname RAMP",lipid_sel="resname RAMP POPE POPG",start=0,stop=-1,step=1,verbose=True,no_bins=200,stat="mean"):
        self.tpr=tpr
        self.traj=traj
        self.phosphates=phosphates
        self.PL_sel=PL_sel
        self.RAMP_sel=RAMP_sel
        self.lipid_sel=lipid_sel
        self.start=start
        self.stop=stop
        self.step=step
        self.verbose=verbose
        self.universe=mda.Universe(tpr,traj)
        self.no_bins=no_bins
        self.stat=stat
    
    def order_parameter_calc(self):
        """
        Calculate order parameters and create 2D spatial map.
        
        Workflow:
        1. Assign lipids to leaflets using lipyphilic
        2. Calculate SCC for phospholipid tails (sn-1 and sn-2)
        3. Calculate weighted average of phospholipid tail order parameters
        4. Calculate SCC for all 6 RAMP tails
        5. Calculate weighted average of RAMP tail order parameters
        6. Create 2D spatial map combining both lipid types
        
        """
        # Assign lipids to leaflets.
        self.leaflets=AssignLeaflets(universe=self.universe,lipid_sel=self.phosphates)
        self.leaflets.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)

        # Filter for phospholipids and create upper leaflet mask
        self.phospholipid_leaflets=self.leaflets.filter_leaflets(lipid_sel=self.PL_sel)
        self.phospholipid_leaflet_mask= (self.phospholipid_leaflets == 1)

        # Calculate SCC for phospholipid tails
        self.scc_PL_sn1=SCC(universe=self.universe,tail_sel="resname POPE POPG and name ??A") 
        self.scc_PL_sn2=SCC(universe=self.universe,tail_sel="resname POPE POPG and name ??B")
        self.scc_PL_sn1.run(start=self.start,stop=self.stop,step=self.step)
        self.scc_PL_sn2.run(start=self.start,stop=self.stop,step=self.step)

         # Average over both phospholipid tails
        self.scc_PL_ave=SCC.weighted_average(sn1_scc=self.scc_PL_sn1,sn2_scc=self.scc_PL_sn2)

        # Calculate SCC for RaLPS tails
        self.RAMP_leaflets=self.leaflets.filter_leaflets(lipid_sel=self.RAMP_sel)
        self.scc_RAMP_sn1=SCC(universe=self.universe,tail_sel="resname RAMP and name ??A")
        self.scc_RAMP_sn2=SCC(universe=self.universe,tail_sel="resname RAMP and name ??B")
        self.scc_RAMP_sn3=SCC(universe=self.universe,tail_sel="resname RAMP and name ??C")
        self.scc_RAMP_sn4=SCC(universe=self.universe,tail_sel="resname RAMP and name ??D")
        self.scc_RAMP_sn5=SCC(universe=self.universe,tail_sel="resname RAMP and name ??E")
        self.scc_RAMP_sn6=SCC(universe=self.universe,tail_sel="resname RAMP and name ??F")
        self.scc_RAMP_sn1.run(start=self.start,stop=self.stop,step=self.step)
        self.scc_RAMP_sn2.run(start=self.start,stop=self.stop,step=self.step)
        self.scc_RAMP_sn3.run(start=self.start,stop=self.stop,step=self.step)
        self.scc_RAMP_sn4.run(start=self.start,stop=self.stop,step=self.step)
        self.scc_RAMP_sn5.run(start=self.start,stop=self.stop,step=self.step)
        self.scc_RAMP_sn6.run(start=self.start,stop=self.stop,step=self.step)
        
        # Calculate weighted average of RaLPS tail order parameters.

        self.scc_RAMP_ave=weighted_average_6(sn1_scc=self.scc_RAMP_sn1,sn2_scc=self.scc_RAMP_sn2,sn3_scc=self.scc_RAMP_sn3,sn4_scc=self.scc_RAMP_sn4,sn5_scc=self.scc_RAMP_sn5,sn6_scc=self.scc_RAMP_sn6)
        # Create 2D spatial map of order parameters.
        self.mean,self.mean_tiled,self.x_edges,self.y_edges=Custom_binning_function(scc_object1=self.scc_RAMP_ave,scc_object2=self.scc_PL_ave,lipid_sel=self.lipid_sel,startframe=self.start,endframe=self.stop,stepframe=self.step,bins=self.no_bins,statistic=self.stat,filterby=self.phospholipid_leaflet_mask)

    