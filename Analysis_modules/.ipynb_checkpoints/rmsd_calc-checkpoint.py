import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import pickle

def pickle_files(filename,variable):
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

def Write_out_sel(tpr,traj,sel,xtcname,groname):
    un=mda.Universe(tpr,traj)
    Sel=un.select_atoms(sel)
    Subsel_xtc=f"../Trajectories_and_tprs/{xtcname}"
    Subsel_gro=f"../Trajectories_and_tprs/{groname}"
    Sel.write(f"{Subsel_xtc}",frames='all')
    Sel.write(f"{Subsel_gro}",frame=-1)
    return Subsel_xtc,Subsel_gro
    
    
class RMSD_calculator:
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
        # Sort out tpr file here - add subtpr 
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
        ref=mda.Universe(f"../Trajectories_and_tprs/{self.tpr}",f"../Trajectories_and_tprs/{self.traj}")
        average_protein=align.AverageStructure(ref,ref,select=f"protein and backbone",ref_frame=self.refframe).run(verbose=self.verbose,start=self.start,stop=self.stop)
        ref_new=average_protein.universe
        aligner=align.AlignTraj(ref,ref_new,select=f"protein and backbone",in_memory=False,filename=f'../Trajectories_and_tprs/aligned_{self.selname}.xtc').run(verbose=self.verbose,start=self.start,stop=self.stop)
        u=mda.Universe(f"../Trajectories_and_tprs/{self.tpr}",f"../Trajectories_and_tprs/aligned_{self.selname}.xtc")
        backbone=u.select_atoms(f"protein and backbone")
        R=rms.RMSF(backbone).run(verbose=self.verbose)

        pickle_files(f"{self.savingfolder}{self.selname}_RMSF",[backbone.resids,R.rmsf])
        return backbone.resids,R.rmsf
       
    
