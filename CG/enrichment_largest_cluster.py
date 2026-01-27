"""
Module for calculating lipid-lipid enrichment indices and identifying largest lipid cluster in MD simulations of membrane systems using the Lipyphilic and MDAnalysis Python packages.
"""

import pathlib
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from lipyphilic.lib.assign_leaflets import AssignLeaflets
from lipyphilic.lib.neighbours import Neighbours


class Enrichment_largest_cluster:
     """
    Calculates lipid enrichment indices and largest cluster in membrane simulations.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    step : int
        Analyze every nth frame.
    saving_folder : str
        Directory path where results will be saved.
    lipid_sel : str, default="name PO1 PO2 PO4"
        MDAnalysis selection string for lipid headgroup atoms/beads.
        Default selection works for MARTINI force field.
    start : int, optional
        First frame to analyze. If None, starts from beginning.
    stop : int, optional
        Last frame to analyze. If None, analyzes to end.
    verbose : bool, default=True
        Print progress information during analysis.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        Universe formed by tpr and traj.
    savingfolder : str
        Directory for saving output files.
    
    Notes
    -----
    - Results are automatically saved to text files
    """
    def __init__(self,tpr,traj,step,saving_folder,lipid_sel="name PO1 PO2 PO4",start=None,stop=None,verbose=True):
        self.tpr=tpr
        self.traj=traj
        self.lipid_sel="name PO1 PO2 PO4"
        self.start=start
        self.stop=stop
        self.step=step
        self.verbose=verbose
        self.universe=mda.Universe(tpr,traj,continuous=True)
        self.savingfolder=saving_folder


    def Enrichment_calculation(self,cutoff=12.0, return_enrichment=True):
         """
        Computes the lipid enrichment indices for each lipid species. Code adapted from 
        Lipyphilic tutorials.
        
        Parameters
        ----------
        cutoff : float, default=12.0
            Distance cutoff (Å) for defining lipid neighbors.
        return_enrichment : bool, default=True
            Whether to calculate enrichment indices (True) or just counts (False).
            
        Returns
        -------
        unique_labels : ndarray
            Array of unique lipid species labels found in the system.
        enrichment : pandas.DataFrame
            DataFrame containing enrichment indices for all lipid pairs.
            Columns include 'Label', 'Frame', 'Time', and 'fe{lipid}' for
            each lipid species.
        neighbours : lipyphilic.lib.neighbours.Neighbours
            The Neighbours analysis object containing all neighbor data.
            
        Side Effects
        ------------
        For each lipid species, saves a text file at:
        {saving_folder}{lipid_label}enrichment.dat
        containing [Time (μs), enrichment_index] for that lipid species.
        
        Notes
        -----
        - Time is converted from ps to μs using conversion factor 1e-6,
        - For each lipid species, saves a text file as: 
        {saving_folder}{lipid_label}enrichment.dat containing [Time (μs), enrichment_index] 
        for that lipid species.
        """

        # Calculates lipid neighbours
        neighbours=Neighbours(universe=self.universe,lipid_sel=self.lipid_sel,cutoff=cutoff)
        neighbours.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)
        counts=neighbours.count_neighbours()

        # Count neighbours and calculate enrichment
        counts, enrichment = neighbours.count_neighbours(return_enrichment=return_enrichment)
        unique_labels = counts.Label.unique()

        # Save enrichment data for each lipid species
        for ref in unique_labels:
            species_fe = enrichment.loc[enrichment.Label == ref].copy()
            ps_to_microseconds = 1e-6
            species_fe["Time"] = species_fe["Frame"].values * neighbours._trajectory.dt * ps_to_microseconds
    
            for l in unique_labels:
                if ref == l:
                    np.savetxt(str(self.savingfolder)+str(l)+"enrichment"+".dat",            np.c_[species_fe["Time"],species_fe["fe"+str(l)]])
        return unique_labels, enrichment, neighbours


    def Cluster_calculations(self,neighbours,POPG=True,RAMP=True,POPE=True):
         """        
        Identifies lipid clusters (groups of neighboring lipids of the same type)
        and tracks the size of the largest cluster for each specified lipid species.
        Leaflet assignment is performed to analyze upper and lower leaflets separately.
        Adapted from code in Lipyphilic tutorials.
        
        Parameters
        ----------
        neighbours : lipyphilic.lib.neighbours.Neighbours
            Neighbours analysis object from Enrichment_calculation().
        POPG : bool, default=True
            Calculate largest cluster for POPG lipids in upper leaflet.
        RAMP : bool, default=True
            Calculate largest cluster for RaLPS lipids (both leaflets).
        POPE : bool, default=True
            Calculate largest cluster for POPE lipids in upper leaflet.
            
        Returns
        -------
        Times : ndarray
            Time points in microseconds.
        largest_cluster_RAMP : ndarray
            Size of largest RaLPS cluster at each time point (if RAMP=True).
        largest_cluster_upper_POPE : ndarray
            Size of largest POPE cluster in upper leaflet at each time point
            (if POPE=True).
        largest_cluster_upper_POPG : ndarray or int
            Size of largest POPG cluster in upper leaflet at each time point
            (if POPG=True), otherwise returns 0.
            
        Notes
        -----
        - Leaflet assignment: upper leaflet (leaflets == 1), lower leaflet (leaflets == -1)
        - Cluster size represents number of lipids in the largest connected component
        Saves cluster size data to text files:
        - {saving_folder}RAMP_cluster.dat: [Time (ps), cluster_size]
        - {saving_folder}POPE_cluster.dat: [Time (ps), cluster_size]
        - {saving_folder}POPG_cluster.dat: [Time (ps), cluster_size]
        
        Warnings
        --------
        The time calculation contains hardcoded values (800, 1000000, 5e-5)
        that may need adjustment for different simulation parameters.
        """

        # Assign lipids to leaflets
        leaflets=AssignLeaflets(universe=self.universe,lipid_sel=self.lipid_sel)
        leaflets.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)
    
        membrane = self.universe.select_atoms(self.lipid_sel).residues

         # Create leaflet masks
        upper_leaflet_mask = leaflets.leaflets == 1
        lower_leaflet_mask = leaflets.leaflets == -1

        # Initialize time arrays
        Times=[]
        Timesps=[]
        ps_to_microseconds = 5e-5
        
        # Calculate RAMP clusters (both leaflets)
        if RAMP==True:
            largest_cluster_RAMP = neighbours.largest_cluster(cluster_sel="resname RAMP")
            for index, value in enumerate(largest_cluster_RAMP):
                Times.append((index+1)*(800)/(1000000))
            for x in Times:
                y=x*neighbours._trajectory.dt/ps_to_microseconds
                Timesps.append(y)
            np.savetxt(str(self.savingfolder)+"RAMP_cluster.dat", np.c_[Timesps,largest_cluster_RAMP])

        # Calculate POPE clusters (upper leaflet only)
        if POPE==True:
            largest_cluster_upper_POPE = neighbours.largest_cluster(cluster_sel='resname POPE', filter_by=upper_leaflet_mask)
            np.savetxt(str(self.savingfolder)+"POPE_cluster.dat", np.c_[Timesps,largest_cluster_upper_POPE])

        # Calculate POPG clusters (upper leaflet only)
        if POPG==True:
            largest_cluster_upper_POPG = neighbours.largest_cluster(cluster_sel='resname POPG', filter_by=upper_leaflet_mask)
            np.savetxt(str(self.savingfolder)+"POPG_cluster.dat", np.c_[Timesps,largest_cluster_upper_POPG])
        else:
            largest_cluster_upper_POPG=0
    
      
        return np.array(Times), largest_cluster_RAMP, largest_cluster_upper_POPE, largest_cluster_upper_POPG