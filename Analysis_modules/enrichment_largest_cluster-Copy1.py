import pathlib
import pickle

import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder

from lipyphilic.lib.assign_leaflets import AssignLeaflets
from lipyphilic.lib.neighbours import Neighbours

# Maybe worth object orienting this program - making a class and then adding the two functions under the class

class Enrichment_largest_cluster:
    
    def __init__(self,tpr,traj,step,saving_folder,lipid_sel="name PO1 PO2 PO4",start=None,stop=None,verbose=True):
        self.tpr=tpr
        self.traj=traj
        self.lipid_sel="name PO1 PO2 PO4"
        self.start=start
        self.stop=stop
        self.step=step
        self.verbose=verbose
        self.universe=mda.Universe(tpr,traj)
        self.savingfolder=saving_folder


    def Enrichment_calculation(self,cutoff=12.0, return_enrichment=True):
        neighbours=Neighbours(universe=self.universe,lipid_sel=self.lipid_sel,cutoff=cutoff)
        neighbours.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)
        counts=neighbours.count_neighbours()
        counts, enrichment = neighbours.count_neighbours(return_enrichment=return_enrichment)
        unique_labels = counts.Label.unique()
    
        for ref in unique_labels:
            species_fe = enrichment.loc[enrichment.Label == ref].copy()
            ps_to_microseconds = 1e-6
            species_fe["Time"] = species_fe["Frame"].values * neighbours._trajectory.dt * ps_to_microseconds
    
            for l in unique_labels:
                if ref == l:
                    np.savetxt(str(self.savingfolder)+str(l)+"enrichment"+".dat",            np.c_[species_fe["Time"],species_fe["fe"+str(l)]])
        return unique_labels, enrichment, neighbours


    def Cluster_calculations(self,neighbours,POPG=True,RAMP=True,POPE=True):
        
        leaflets=AssignLeaflets(universe=self.universe,lipid_sel=self.lipid_sel)
        leaflets.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)
    
        membrane = self.universe.select_atoms(self.lipid_sel).residues

        upper_leaflet_mask = leaflets.leaflets == 1
        lower_leaflet_mask = leaflets.leaflets == -1
        
        Times=[]
        Timesps=[]
        ps_to_microseconds = 5e-5
        
        
        if RAMP==True:
            largest_cluster_RAMP = neighbours.largest_cluster(cluster_sel="resname RAMP")
            for index, value in enumerate(largest_cluster_RAMP):
                Times.append((index+1)*(800)/(1000000))
            for x in Times:
                y=x*neighbours._trajectory.dt/ps_to_microseconds
                Timesps.append(y)
            np.savetxt(str(self.savingfolder)+"RAMP_cluster.dat", np.c_[Timesps,largest_cluster_RAMP])
      
        if POPE==True:
            largest_cluster_upper_POPE = neighbours.largest_cluster(cluster_sel='resname POPE', filter_by=upper_leaflet_mask)
            np.savetxt(str(self.savingfolder)+"POPE_cluster.dat", np.c_[Timesps,largest_cluster_upper_POPE])

        
        if POPG==True:
            largest_cluster_upper_POPG = neighbours.largest_cluster(cluster_sel='resname POPG', filter_by=upper_leaflet_mask)
            np.savetxt(str(self.savingfolder)+"POPG_cluster.dat", np.c_[Timesps,largest_cluster_upper_POPG])
        else:
            largest_cluster_upper_POPG=0
    
      
        return np.array(Times), largest_cluster_RAMP, largest_cluster_upper_POPE, largest_cluster_upper_POPG