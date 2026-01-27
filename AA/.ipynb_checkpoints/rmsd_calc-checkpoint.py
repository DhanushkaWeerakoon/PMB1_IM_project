"""
Module for calculating Root Mean Square Deviation (RMSD) and Root Mean Square Fluctuation (RMSF) using MDAnalysis, with automatic result saving.
"""

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import pickle

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

def Write_out_sel(tpr,traj,sel,xtcname,groname):
    """
    Create a subtrajectory and substructure (xtcname, groname) from a trajectory and structure 
    (traj, tpr) using an atom selection (sel).
            
    Parameters
    ----------
    tpr : str
        Path to input topology file.
    traj : str
        Path to input trajectory file.
    sel : str
        MDAnalysis atom selection string.
    xtcname : str
        Output filename for trajectory (e.g., 'protein.xtc').
    groname : str
        Output filename for structure (e.g., 'protein.gro').
        
    Returns
    -------
    Subsel_xtc : str
        Full path to output trajectory file 
        ('../Trajectories_and_tprs/{xtcname}').
    Subsel_gro : str
        Full path to output structure file
        ('../Trajectories_and_tprs/{groname}').
    """

    
    un=mda.Universe(tpr,traj)
    Sel=un.select_atoms(sel)
    Subsel_xtc=f"../Trajectories_and_tprs/{xtcname}"
    Subsel_gro=f"../Trajectories_and_tprs/{groname}"
    Sel.write(f"{Subsel_xtc}",frames='all')
    Sel.write(f"{Subsel_gro}",frame=-1)
    return Subsel_xtc,Subsel_gro
    
    
class RMSD_calculator:
    """
        This class creates a subset of the input trajectory and topology file, then 
        aligns the subtrajectory to its initial frame. Following this, it calculates the 
        RMSD relative to the subtrajectory initial frame via the RMSD_calculation function.
        Results are automatically saved to pickle files.
    
        Parameters
        ----------
        tpr : str
            Path to topology file (e.g., .tpr, .gro).
        traj : str
            Path to trajectory file (e.g., .xtc, .trr).
        saving_folder : str
            Directory path where RMSD results will be saved.
        sel : str
            MDAnalysis selection string for subtrajectory selection and RMSD calculation 
            (also used for alignment if sel2 is not provided).
        selname : str
            Base name for output files and subtrajectory.
        verbose : bool, default=True
            Print progress information during analysis.
        sel2 : str, optional
            MDAnalysis selection string for alignment.            
        start : int, default=0
            First frame to analyze.
        stop : int, default=-1
            Last frame to analyze (-1 for all frames).
        step : int, default=1
            Analyze every nth frame.
        
        Attributes
        ----------
        All parameters.
        subtraj : str
            Path to created subtrajectory file.
        subtpr : str
            Path to created substructure file.
        
   
        Notes
        -----
        - RMSD is calculated with centering and superposition enabled
        - Reference is the first frame (frame 0) of the subtrajectory
        """
    def __init__(self,tpr,traj,saving_folder,sel,selname,verbose=True,sel2=None,ref_frame=None,start=0,stop=-1,step=1):
        self.tpr=tpr
        self.traj=traj
        self.savingfolder=saving_folder
        self.sel=sel
        self.sel2=sel2
        self.selname=selname
        self.verbose=verbose
        self.refframe=ref_frame
        self.start=start
        self.stop=stop
        self.step=step
        
        
        self.subtraj,self.subtpr=Write_out_sel(tpr=self.tpr,traj=self.traj,sel=self.sel,xtcname=f"{self.selname}.xtc",groname=f"{self.selname}.gro")
        
    def RMSD_calculation(self):
        """
        Calculate RMSD for the trajectory.
        
        Aligns structures using the selection, and computes RMSD. The reference frame is the
        first frame (index 0).
        
        Returns
        -------
        rmsd : ndarray
            Transposed RMSD array with shape (4, n_frames):
            - Row 0: Frame number
            - Row 1: Time (ps)
            - Row 2: RMSD (Å)
            - Row 3: RMSD (Å) - duplicate from MDAnalysis output
            
        Notes
        ------------
        Saves RMSD results to pickle file at: {saving_folder}{selname}
        Aligned reference and aligned trajectory are saved in '../Trajectories_and_tprs'
        
        """

        
        aligned=mda.Universe(f"../Trajectories_and_tprs/{self.selname}.gro",f"{self.subtraj}")
        aligned_ref=mda.Universe(f"../Trajectories_and_tprs/{self.selname}.gro",f"{self.subtraj}")
        
        aligned.trajectory[-1]
        aligned_ref.trajectory[0]
        
        if self.sel2 != None:
            R=mda.analysis.rms.RMSD(aligned,aligned_ref,select=self.sel2,center=True,superpose=True)
        else:
            R=mda.analysis.rms.RMSD(aligned,aligned_ref,select=self.sel,center=True,superpose=True)
        
        R.run(verbose=self.verbose)
        rmsd=R.rmsd.T
        
        pickle_files(f"{self.savingfolder}{self.selname}",rmsd)
        return rmsd
        
    
class RMSF_calculator:
    """
    Computes per-residue RMSF of proteins over trajectory.
    
    Calculates the average structure computed from all frames.
    trajectory, aligning the trajectory to the average structure, then calculating RMSF of 
    backbone atoms.
    
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro) in '../Trajectories_and_tprs/'.
    traj : str
        Path to trajectory file (e.g., .xtc, .trr) in '../Trajectories_and_tprs/'.
    saving_folder : str
        Directory path where RMSF results will be saved.
    selname : str
        Base name for output files.
    ref_frame : int
        Reference frame index for creating average structure.
    start : int, default=0
        First frame to analyze.
    stop : int, default=-1
        Last frame to analyze (-1 for all frames).
    step : int, default=1
        Analyze every nth frame.
    verbose : bool, default=True
        Print progress information during analysis.
    """
    
    def __init__(self,tpr,traj,saving_folder,selname,ref_frame,start=0,stop=-1,step=1,verbose=True):
        self.tpr=tpr
        self.traj=traj
        self.savingfolder=saving_folder
        self.verbose=verbose
        self.start=start
        self.stop=stop
        self.step=step
        self.selname=selname
        self.refframe=ref_frame
        
    def RMSF_calculation(self):
        """
        Calculate per-residue RMSF for protein backbone.
        
        Create average structure from protein backbone, aligns trajectory to average 
        structure and calculates RMSF for backbone atoms
        
        Returns
        -------
        resids : ndarray
            Residue IDs for backbone atoms.
        rmsf : ndarray
            RMSF values (Å) for each residue.

          
        Notes
        -----
        - Creates aligned trajectory at:
          '../Trajectories_and_tprs/aligned_{selname}.xtc'
        - Saves results to pickle file at:
          '{saving_folder}{selname}_RMSF' containing [resids, rmsf]
        """

        

        # Load trajectory
        ref=mda.Universe(f"../Trajectories_and_tprs/{self.tpr}",f"../Trajectories_and_tprs/{self.traj}")

        # Create average protein structure using protein backbone atoms
        average_protein=align.AverageStructure(ref,ref,select=f"protein and backbone",ref_frame=self.refframe).run(verbose=self.verbose,start=self.start,stop=self.stop)
        ref_new=average_protein.universe

        # Align trajectory to average structure
        aligner=align.AlignTraj(ref,ref_new,select=f"protein and backbone",in_memory=False,filename=f'../Trajectories_and_tprs/aligned_{self.selname}.xtc').run(verbose=self.verbose,start=self.start,stop=self.stop)

        # Load aligned trajectory and calculate RMSF of protein backbone atoms
        u=mda.Universe(f"../Trajectories_and_tprs/{self.tpr}",f"../Trajectories_and_tprs/aligned_{self.selname}.xtc")
        backbone=u.select_atoms(f"protein and backbone")
        R=rms.RMSF(backbone).run(verbose=self.verbose)

        pickle_files(f"{self.savingfolder}{self.selname}_RMSF",[backbone.resids,R.rmsf])
        return backbone.resids,R.rmsf
       
    
