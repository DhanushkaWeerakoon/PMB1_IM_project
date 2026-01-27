import MDAnalysis as mda
from MDAnalysis.analysis import contacts
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.base import AnalysisFromFunction
import numpy as np
import pickle 
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)

def pickle_files(filename,variable):
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

def contacts_within_cutoff(u, group_a, group_b, radius=6.0):
    # calculate distances between group_a and group_b
    dist = contacts.distance_array(group_a.positions, group_b.positions,box=u.dimensions)
    # determine which distances <= radius
    n_contacts = contacts.contact_matrix(dist, radius).sum()
    return n_contacts

Calculate_contacts = analysis_class(contacts_within_cutoff)

class Intermolecular_contacts():
    def __init__(self,tpr,traj,saving_folder,selA,selB,selname,phosphates='name PO1 PO2 PO4',proximity=6.0,start=0,stop=-1,step=1,verbose=True,update=False):
        self.tpr=tpr
        self.traj=traj
        self.savingfolder=saving_folder
        self.universe=mda.Universe(tpr,traj)
        self.selA=selA
        self.selB=selB
        self.selname=selname
        self.phosphates=phosphates
        self.proximity=proximity
        self.start=start
        self.stop=stop
        self.step=step
        self.verbose=verbose
        self.update=update
        
    def Contact_calculator(self):
        
        L=LeafletFinder(self.universe,self.phosphates,pbc=True)
        leaflet0=L.groups(0)
        leaflet1=L.groups(1)
        
        A=self.universe.select_atoms(self.selA)
        
        if self.selB=="resname POPE" or self.selB=="resname POPG":
            print(self.selB)
            B=leaflet0.residues.atoms.select_atoms(self.selB,updating=self.update)
        else:
            B=self.universe.select_atoms(self.selB,updating=self.update)
        print(len(B.residues))
            
        Contacts=Calculate_contacts(u=self.universe,group_a=A,group_b=B,radius=self.proximity)
        Contacts.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)
        
        Results=Contacts.results["timeseries"]
        Times=Contacts.results["times"]
        
        pickle_files(f"{self.savingfolder}{self.selname}",[Times,Results])
        
        
        return Times,Results
        
   
        
        

        