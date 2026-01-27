import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
import MDAnalysis.lib.pkdtree as pkdtree


class Occupancy(AnalysisBase):
    def __init__(self,tpr,traj,selA,selB,offset=0,protein_residues=417,proximity=6.0,start=0,stop=-1,step=1):
        universe=mda.Universe(tpr,traj,continuous=True)
        super(Occupancy,self).__init__(universe.trajectory)
        self.tpr=tpr
        self.traj=traj
        self.universe=universe
        self.selA=selA
        self.selB=selB
        self.protein_residues=protein_residues
        self.proximity=proximity
        self.offset=offset

        self.Selection_A=self.universe.select_atoms(f"{self.selA}")
        self.Selection_B=self.universe.select_atoms(f"{self.selB}")
        self.Residue_list=[self.universe.select_atoms(f"{self.selB} and resid {i+1+self.offset}") for i in range(self.protein_residues)]

    def _prepare(self):
        self.Contacts_over_time=[]

    def search(self,selA,selB,cutoff,level="A"):
        
        positions=selB.atoms.position

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
        resname_hist={}

        for i in range(self.protein_residues):
            resname_hist[i+1+self.offset]=0
            self.kdtree=pkdtree.PeriodicKDTree(box=self.universe.dimensions)
        self.kdtree.set_coords(self.Selection_A.positions,cutoff=self.proximity+0.1)
        for (i,j) in zip(self.Residue_list,range(self.protein_residues)):
            for atom in i:
                near=self.search(selA=self.Selection_A,selB=atom,cutoff=self.proximity, level="A")
                if len(near) ==0:
                    continue
                else:
                    resname_hist[j+1+self.offset]=1
                    break
        
        residues=[j for i,j in resname_hist.items()]
        self.Contacts_over_time.append(residues)
   
        #for i in range(self.protein_residues):
         #   resname_hist[i+1+self.offset]=0
            
        #for residue in Selection_B.residues:
         #   print(residue)
         #   print(residue.resid)

            
            #near=self.search(selA=self.selectionA,sel=residue,cutoff=self.proximity,level="A")
          #  if len(near) == 0:
          #      continue
          #  else:
          #      continue
                #resids=[x.resid for x in near]
                ##for resid in resids:
                 #   if resid in resname_hist:
                 #       resname_hist[resid]=1
        #self.Contacts_over_time.append(residue_list)
        #residue_list=[j for i,j in resname_hist.items()]
        
    
    def _conclude(self):
        self.Contacts_over_time=np.array(self.Contacts_over_time,dtype='object')


