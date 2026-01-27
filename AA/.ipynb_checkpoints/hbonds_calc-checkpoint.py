"""
Module for calculating intramolacular hydrogen bonds using MDAnalysis, with automatic result
saving.
"""

import pickle
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

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

class Hbonds():

    """

    This class wraps MDAnalysis HydrogenBondAnalysis to compute inter/intramolecular hydrogen
    bonds and their lifetimes from simulation trajectories, automatically saving results to
    pickle files.

    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .pdb).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    saving_folder : str
        Directory path where results will be saved.
    selA : str
        MDAnalysis selection string for first group.
    selname : str
        Base name for output pickle file.
    selB : str, optional
        MDAnalysis selection string for second group. If None, analyzes
        hydrogen bonds within selA only.
    verbose : bool, default=True
        Print progress information during analysis.
    start : int, default=0
        First frame to analyze.
    stop : int, default=-1
        Last frame to analyze (-1 for all frames).
    step : int, default=10
        Analyze every nth frame.
    update : bool, default=True
        Update atom selections each frame (important for dynamic selections).

        
    Attributes
    ----------
    All parameters.
    universe : MDAnalysis.Universe
        The loaded trajectory universe.
    """
    def __init__(self,tpr,traj,saving_folder,selA,selname,selB=None,verbose=True,start=0,stop=-1,step=10,update=True):
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
        self.selname=selname
    
   
    def Hbonds_calculation(self):
        """
        Calculate hydrogen bonds and save results to pickle file.
        
        Returns
        -------
        If selB is None:
            times : ndarray
                Trajectory times in nanoseconds.
            counts : ndarray
                Number of hydrogen bonds at each time point.
                
        If selB is provided:
            times : ndarray
                Trajectory times in nanoseconds.
            total_counts : ndarray
                Total hydrogen bonds (acceptor + donor) at each time point.
            acceptor_counts : ndarray
                Hydrogen bonds where selA is acceptor, selB is donor.
            donor_counts : ndarray
                Hydrogen bonds where selA is donor, selB is acceptor.
     
        Notes
        -----
        Times are converted from ps to ns by dividing by 1000.
        Saves all returned properties in a pickle file located at {savingfolder}{selname}.
        """
        
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
            
           

            
    
                      
            
            
            
        
        
