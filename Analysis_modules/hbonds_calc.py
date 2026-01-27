import pickle
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

def pickle_files(filename,variable):
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

class Hbonds():
    def __init__(self,tpr,traj,saving_folder,selA,selname,selB=None,verbose=True,start=0,stop=-1,step=10,update=True, radius=10):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj)
        self.savingfolder=saving_folder
        self.selA=selA
        self.selB=selB
        self.verbose=verbose
        self.start=start
        self.stop=stop
        self.step=step
        self.update=update
        self.radius=radius
        self.selname=selname
    
   
    def Hbonds_calculation(self):
        if self.selB == None:
            hbonds_container=HydrogenBondAnalysis(universe=self.universe,update_selections=self.update)
            SelA_hydrogens_sel=hbonds_container.guess_hydrogens(self.selA)
            SelA_acceptors_sel=hbonds_container.guess_acceptors(self.selA)
            
            hbonds_container.hydrogen_sel=f"{SelA_hydrogens_sel}"
            hbonds_container.acceptors_sel=f"{SelA_acceptors_sel}"
            
            hbonds_container.run(verbose=self.verbose,step=self.step,start=self.start)  
            
            pickle_files(f"{self.savingfolder}{self.selname}",[hbonds_container.times/1000,hbonds_container.count_by_time()])
        
            return hbonds_container.times/1000,hbonds_container.count_by_time()
        
        elif self.selB != None:  
            hbonds_acceptor=HydrogenBondAnalysis(universe=self.universe,update_selections=self.update)
            hbonds_donor=HydrogenBondAnalysis(universe=self.universe,update_selections=self.update)
            
            SelA_hydrogens_sel=hbonds_donor.guess_hydrogens(self.selA)
            SelA_acceptors_sel=hbonds_acceptor.guess_acceptors(self.selA)
            SelB_hydrogens_sel=hbonds_acceptor.guess_hydrogens(self.selB)
            SelB_acceptors_sel=hbonds_donor.guess_acceptors(self.selB)
            
            hbonds_acceptor.hydrogens_sel=SelB_hydrogens_sel
            hbonds_acceptor.acceptors_sel=SelA_acceptors_sel
            
            hbonds_donor.hydrogens_sel=SelA_hydrogens_sel
            hbonds_donor.acceptors_sel=SelB_acceptors_sel
            
            hbonds_donor.run(verbose=self.verbose,step=self.step,start=self.start)
            hbonds_acceptor.run(verbose=self.verbose,step=self.step,start=self.start)
            
            hbonds_total=np.vstack([hbonds_acceptor.count_by_time(),hbonds_donor.count_by_time()])
            hbonds_total_sum=np.sum(hbonds_total,axis=0)
            
            pickle_files(f"{self.savingfolder}{self.selname}",[hbonds_acceptor.times/1000,hbonds_total_sum,hbonds_acceptor.count_by_time(),hbonds_donor.count_by_time()])
        return hbonds_acceptor.times/1000,hbonds_total_sum,hbonds_acceptor.count_by_time(),hbonds_donor.count_by_time()
            
           

            
    
                      
            
            
            
        
        
