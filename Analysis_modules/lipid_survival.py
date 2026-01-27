import numpy as np
import tqdm
import MDAnalysis as mda
import MDAnalysis.lib.pkdtree as pkdtree
from MDAnalysis.core.groups import Atom, AtomGroup
import argparse
from numba import jit

# coding=utf-8


#tqdm - module to make loops show a progress metre. Just use as follows:
#from tqdm import tqdm 
    #for i in tqdm(range(10000)):
        #...
        
#KDTree is a Scipy package for quick nearest-neighbour lookup. Provides an index into a set of k-dimensional points which can be used to rapidly look up nearest neighbours at any point. 

#Mda provides wrapped to allows searches on a KDTree involving PBC.

#classMDAnalysis.lib.pkdtree.PeriodicKDTree(box=None, leafsize=10)

#Parameters:
#- box
#- leafsize - number of entries in leaves of KDTree

#Numba translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN.

def distance_sq(pos1, pos2):
    diff = pos2 - pos1
    return np.sum(np.square(diff))

@jit(nopython=True)
def survival_prob(lipids, dt):
    bound = 0
    total = 0
    for i in range(len(lipids)):
        lipid = lipids[i]
        for i in range(len(lipid) - dt):
            if lipid[i] == 1:
                if lipid[i + dt] == 1:
                    bound += 1
                total += 1

    return bound, total

class NeighbourSearchFast(object):
    '''Class to rapility calculate the neighbours of 'atomgroup' with the 'searchgroup'. Note that all atomgroups
     are MDAanalysis objects. Used kdTree to do this.'''
    def __init__(self, atomgroup, box=None):
        # Initialises for lipids - makes universe, box, KDTree etc.
        self.atomgroup = atomgroup
        self._u = atomgroup.universe
        self.box = box
        self.kdtree = pkdtree.PeriodicKDTree(box=self.box)


    def search(self, searchgroup, radius,level="A"):
        self.kdtree.set_coords(self.atomgroup.positions,cutoff=radius + 0.1)
        #print(self.atomgroup.positions)
        #kdtree.set_coords(coords,cutoff=None) - constructs KDTree from the coordinates
        # Wrapping of coordinates to primary unit cell enforced before any distance evaluations.
        # For non-periodic calculations, don't provide cutoff
        if isinstance(searchgroup, Atom):
        # isinstance() function returns True if the specified object is of the specified type
            positions = searchgroup.positions.reshape(1, 3)
        else:
            positions = searchgroup.positions
            
        unique_idx = self.kdtree.search(positions, radius)
        # search(centers,radius) - searches all points within radius from centers and their periodic images
        # All centers coordinates are wrapped around central cell to enable distance evaluations from points in the tree and their images
        
        self.indices = unique_idx
        # Set indices for class
        # Returns unique indices and level
        return self._index2level(unique_idx, level)

    def _index2level(self, indices, level):
        n_atom_list = self.atomgroup[indices]
        if level == 'A':
            if not n_atom_list:
                return []
            else:
                return n_atom_list
        elif level == 'R':
            return list({a.residue for a in n_atom_list})
        elif level == 'S':
            return list(set([a.segment for a in n_atom_list]))
        else:
            raise NotImplementedError('{0}: level not implemented'.format(level))
    

class Lipid_survival():
    def __init__(self,tpr,traj,PMB1_id,saving_folder,save_prefix="lifetime",start=0,stop=-1,step=1,lipid_select="RAMP",protein_select="PMB1",proximity=6.0):
        self.tpr=tpr
        self.traj=traj
        self.prefix=save_prefix
        self.savingfolder=saving_folder
        self.start=start
        self.stop=stop
        self.step=step
        self.lipid_select=lipid_select
        self.protein_select=protein_select
        self.PMB1_id=PMB1_id
        self.universe=mda.Universe(tpr,traj,continuous=True)
        self.proximity=proximity
    
    def Survival_calculation(self):
        if self.lipid_select == "RAMP":
            lipids=self.universe.select_atoms(f"resname {self.lipid_select}")
            print(self.PMB1_id)
            protein=self.universe.select_atoms(f"resname {self.protein_select} and resid {self.PMB1_id}")
        
        elif self.lipid_select == "PMB1":
            lipids=self.universe.select_atoms(f"resname {self.lipid_select} and resid {self.PMB1_id}")
            protein=self.universe.select_atoms(f"{self.protein_select}")
            
        lipids_on_prev=[]
        time_on={}
        
        dt=(self.universe.trajectory[1].time-self.universe.trajectory[0].time)/1000
        self.universe.trajectory[0]
        
        lifetimes_per_resid={}
        
        lipid_frame_contacts=dict([(resid,np.zeros(len(self.universe.trajectory[self.start:self.stop:self.step]))) for resid in lipids.residues.resids])
        
        for i, frame in enumerate(tqdm.tqdm(self.universe.trajectory[self.start:self.stop:self.step])):
            frame_contacts=dict([(resid,0) for resid in lipids.residues.resids])
            tree=NeighbourSearchFast(lipids,box=self.universe.dimensions)
            lipids_near=tree.search(protein,self.proximity,level='R')
            resids_on=[lipid.resid for lipid in lipids_near]
    
    #Make lipids_on_prev if it doesn't exist - then add timepoints to resids_on
            if not lipids_on_prev:
                lipids_on_prev = resids_on
                for resid in resids_on:
                    time_on[resid] = dt
                continue

            for resid in resids_on:
        #Increase count for each frame
                lipid_frame_contacts[resid][i] += 1
        # If lipid was present previously, state it is present in next frame
                if resid in lipids_on_prev:
                    time_on[resid] += dt
                else:
                    time_on[resid] = dt
    
            for resid in lipids_on_prev:
                if resid not in resids_on:
                    try:
                        lifetimes_per_resid[resid].append(time_on[resid])

                    except KeyError:
                        lifetimes_per_resid[resid] = [time_on[resid]]
                    finally:
                        del time_on[resid]
            lipids_on_prev = resids_on
            
            
        for resid in time_on:
            try:
                lifetimes_per_resid[resid].append(time_on[resid])
            except KeyError:
                lifetimes_per_resid[resid] = [time_on[resid]]
                
        lifetimes = {}
        survival_rates = {}
        for resid in lifetimes_per_resid:
            resname = self.universe.select_atoms("resid " + str(resid)).residues[0].resname
            try:
                survival_rates[resname].append(lipid_frame_contacts[resid])
                lifetimes[resname].extend(lifetimes_per_resid[resid])
            except KeyError:
                lifetimes[resname] = lifetimes_per_resid[resid]
                survival_rates[resname] = [lipid_frame_contacts[resid]]
                
                
        for resname in survival_rates:
            survival_rates[resname] = np.array(survival_rates[resname])
        
        return survival_rates, lifetimes
    
    def Write_files(self,survival_rates,lifetimes):
        PMB1_id_change=self.PMB1_id.replace(":","_")
        for resname in survival_rates:
            with open(f"{self.savingfolder}lifetime_{self.lipid_select}_{self.protein_select}_{PMB1_id_change}_survival.dat", "w") as out:
                template = "{0:^6d}{1:^8.3e}"
                
                # Check if this is ok - can't be too sure
                for dt in tqdm.tqdm(range(0, 50000, 1)):
                    bound, total = survival_prob(survival_rates[resname], dt)
                try:
                    prob = bound / total
                except ZeroDivisionError:
                    prob = 0.
                #print(template.format(dt, prob))
                print(template.format(dt, prob), file=out)
    
        with open(f"{self.savingfolder}lifetime_{self.lipid_select}_{self.protein_select}_{PMB1_id_change}_summary.dat", "w") as out:
            for resname, times in lifetimes.items():
                average = np.mean(times)
                maxi= np.max(times)
                error = np.std(times)
                std_error = error / np.sqrt(len(times))
                bootstraps = []
                for n in range(200):
                    weights = np.random.random(len(times))
                    weights = weights / np.sum(weights)
                    resample = np.random.choice(times, len(times), p=weights)
                    resample_av = np.mean(resample)
                    bootstraps.append(resample_av)

                bootstraps = np.array(bootstraps)
                print(len(bootstraps), len(lifetimes))
                print("Lipid {0}: {1:8.3f} +\- {2:8.3f} ns".format(resname, average, std_error))
                print("Bootstrap: {0:8.3f} +\- {1:8.3f} ns".format(np.mean(bootstraps),np.std(bootstraps)))
                print("Lipid max time = {0} ns".format(maxi))
                print("Lipid {0}: {1:8.3f} +\- {2:8.3f} ns".format(resname, average, std_error), file=out)
                print("Bootstrap: {0:8.3f} +\- {1:8.3f} ns".format(np.mean(bootstraps),np.std(bootstraps) , file=out))
                print("Lipid max time = {0} ns".format(maxi),file=out)

