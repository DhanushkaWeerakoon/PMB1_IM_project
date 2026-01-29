import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
import MDAnalysis.lib.pkdtree as pkdtree
from MDAnalysis.analysis.leaflet import LeafletFinder


class Contacts(AnalysisBase):
    def __init__(self,tpr,traj,selA,selB,proximity=6.0):
        universe=mda.Universe(tpr,traj,continuous=True)
        super(Contacts,self).__init__(universe.trajectory)
        self.tpr=tpr
        self.traj=traj
        self.universe=universe
        self.selA=selA
        self.selB=selB
        self.proximity=proximity
        self.phosphates="(resname RAMP and name PA PB) or (resname POPE POPG and (name P))" 

        if self.selA=="resname POPE" or self.selA=="resname POPG" or self.selA=="resname RAMP":
            L=LeafletFinder(self.universe,self.phosphates,pbc=True)
            leaflet0=L.groups(0)
            leaflet1=L.groups(1)
            self.Selection_A=leaflet0.residues.atoms.select_atoms(self.selA)
        else:
            self.Selection_A=self.universe.select_atoms(f"{self.selA}")
        self.Selection_B=self.universe.select_atoms(f"{self.selB}")
   
    def _prepare(self):
        self.Contacts_over_time=[]

    def search(self,selA,selB,cutoff,level="A"):

        positions=selB.position
        unique_idx=self.kdtree.search(positions,cutoff)
        return self._index2level(selection=selA,indices=unique_idx,level=level)

    def _index2level(self,selection,indices,level):
        n_atom_list=selection[indices]
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
            raise NotImplementedError('{0}: level not implemented.'.format(level))

    def _single_frame(self):
        List=[]
        self.kdtree=pkdtree.PeriodicKDTree(box=self.universe.dimensions)
        self.kdtree.set_coords(self.Selection_A.positions,cutoff=self.proximity+0.1)
        
        for atom in self.Selection_B:
            contacts=self.search(selA=self.Selection_A,selB=atom,cutoff=self.proximity,level="A")
            List.append(len(contacts))
        self.Contacts_over_time.append(np.sum(List))
        
        #List=[]
        #self.kdtree=pkdtree.PeriodicKDTree(box=self.universe.dimensions)
        #self.kdtree.set_coords(self.Selection_A.position, cutoff=self.proximity+0.1)

        #for x in self.Selection_B:
            
        #    positions=x.position
        #print(len(self.indices))
        #    self.indices=self.kdtree.search(positions,self.proximity)
        #    List.append(len(self.indices))
        
        #self.Contacts_over_time.append(np.sum(List))
    
    def _conclude(self):
        self.Contacts_over_time=np.array(self.Contacts_over_time)

