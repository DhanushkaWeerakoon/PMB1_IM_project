import MDAnalysis as mda
import numpy as np
import scipy

from lipyphilic.lib.order_parameter import SCC
from MDAnalysis.analysis.leaflet import LeafletFinder
from lipyphilic.lib.assign_leaflets import AssignLeaflets, AssignCurvedLeaflets

import pickle

from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)

def Time_positions(u,Upper_leaflet,Upper_PMB1s_sel,Lower_PMB1s_sel,Protein_sel):
    LPS_phosphate_coords=[]
    POPE_phosphate_coords=[]
    POPG_phosphate_coords=[]
    PMB1_upper_coords=[]
    PMB1_lower_coords=[]
    Protein_coords=[]
        
    #L = LeafletFinder(universe, 'name PO1 PO2 PO4',pbc=True)
   # Upper_leaflet = L.groups(0)
    #Lower_leaflet = L.groups(1)
    
    LPS_phosphates_upper=Upper_leaflet.select_atoms('name PO1 PO2')
    POPE_phosphates_upper=Upper_leaflet.select_atoms('resname POPE and name PO4')
    POPG_phosphates_upper=Upper_leaflet.select_atoms('resname POPG and name PO4')
    
        
    LPS_phosphate_coords.append(LPS_phosphates_upper.positions)
    POPE_phosphate_coords.append(POPE_phosphates_upper.positions)
    POPG_phosphate_coords.append(POPG_phosphates_upper.positions)
    
    if Upper_PMB1s_sel !=None:
        Upper_PMB1s=u.select_atoms(Upper_PMB1s_sel)
        PMB1_upper_coords.append(Upper_PMB1s.positions)
    
    if Lower_PMB1s_sel != None:
        Lower_PMB1s=u.select_atoms(Lower_PMB1s_sel)
        PMB1_lower_coords.append(Lower_PMB1s.positions)
    
    if Protein_sel != None:
        Protein=u.select_atoms(Protein_sel)
        Protein_coords.append(Protein.positions)
    return LPS_phosphate_coords,POPE_phosphate_coords,POPG_phosphate_coords,PMB1_upper_coords,PMB1_lower_coords,Protein_coords






Calculate_Time_positions = analysis_class(Time_positions)

def run_SCC_calc(universe,tail_sel,start,stop,step):
    x=SCC(universe=universe,tail_sel=tail_sel)
    x.run(start=start,stop=stop,step=step,verbose=True)
    return x

def weighted_average_6(sn1_scc,sn2_scc,sn3_scc,sn4_scc,sn5_scc,sn6_scc):
    sn1_resindices = sn1_scc.tails.residues.resindices
    sn2_resindices = sn2_scc.tails.residues.resindices
    sn3_resindices = sn3_scc.tails.residues.resindices
    sn4_resindices = sn4_scc.tails.residues.resindices
    sn5_resindices = sn5_scc.tails.residues.resindices
    sn6_resindices = sn6_scc.tails.residues.resindices
    
    
    combined_resindices = np.unique(np.hstack([sn1_resindices, sn2_resindices, sn3_resindices,sn4_resindices,sn5_resindices,sn6_resindices]))
    n_residues = combined_resindices.size
               
    scc = np.zeros((n_residues, sn1_scc.n_frames))
    
        
    for species in np.unique(np.hstack([sn1_scc.tails.resnames, sn2_scc.tails.resnames,sn3_scc.tails.resnames,sn4_scc.tails.resnames,sn5_scc.tails.resnames,sn6_scc.tails.resnames])):
        sn1_species_scc = sn1_scc.SCC[sn1_scc.tail_residue_mask[species]]
        sn1_n_atoms_per_lipid = sn1_scc.tails[sn1_scc.tail_atom_mask[species]].n_atoms / len(sn1_species_scc)
                
        sn2_species_scc = sn2_scc.SCC[sn2_scc.tail_residue_mask[species]]
        sn2_n_atoms_per_lipid = sn2_scc.tails[sn2_scc.tail_atom_mask[species]].n_atoms / len(sn2_species_scc)
               
        sn3_species_scc = sn3_scc.SCC[sn3_scc.tail_residue_mask[species]]
        sn3_n_atoms_per_lipid = sn3_scc.tails[sn3_scc.tail_atom_mask[species]].n_atoms / len(sn3_species_scc)
                
        sn4_species_scc = sn4_scc.SCC[sn4_scc.tail_residue_mask[species]]
        sn4_n_atoms_per_lipid = sn4_scc.tails[sn4_scc.tail_atom_mask[species]].n_atoms / len(sn4_species_scc)
                
        sn5_species_scc = sn5_scc.SCC[sn5_scc.tail_residue_mask[species]]
        sn5_n_atoms_per_lipid = sn5_scc.tails[sn5_scc.tail_atom_mask[species]].n_atoms / len(sn5_species_scc)
                
        sn6_species_scc = sn6_scc.SCC[sn6_scc.tail_residue_mask[species]]
        sn6_n_atoms_per_lipid = sn6_scc.tails[sn6_scc.tail_atom_mask[species]].n_atoms / len(sn6_species_scc)
                
        
        species_scc = np.average(
                    np.array([sn1_species_scc, sn2_species_scc,sn3_species_scc,sn4_species_scc,sn5_species_scc,sn6_species_scc]),
                    axis=0,
                    weights=[sn1_n_atoms_per_lipid - 1, sn2_n_atoms_per_lipid - 1,sn3_n_atoms_per_lipid - 1,sn4_n_atoms_per_lipid - 1,sn5_n_atoms_per_lipid - 1,sn6_n_atoms_per_lipid - 1]  # - 1 to obain the number of C-C bonds
                )
                
        species_resindices = np.in1d(combined_resindices, sn1_resindices[sn1_scc.tail_residue_mask[species]])
        scc[species_resindices] = species_scc    
    
    
    sn1_atom_indices = sn1_scc.tails.indices
    sn2_atom_indices = sn2_scc.tails.indices
    sn3_atom_indices = sn3_scc.tails.indices
    sn4_atom_indices = sn4_scc.tails.indices
    sn5_atom_indices = sn5_scc.tails.indices
    sn6_atom_indices = sn6_scc.tails.indices
    
    combined_atom_indices = np.unique(np.hstack([sn1_atom_indices, sn2_atom_indices,sn3_atom_indices,sn4_atom_indices,sn5_atom_indices,sn6_atom_indices]))
        
    new_scc = SCC(universe=sn1_scc.u,
          tail_sel=f"index {' '.join(combined_atom_indices.astype(str))}",)
        
    new_scc.start, new_scc.stop, new_scc.step = sn1_scc.start, sn1_scc.stop, sn1_scc.step
    new_scc.frames = np.arange(new_scc.start, new_scc.stop, new_scc.step)
    new_scc.n_frames = new_scc.frames.size
    new_scc.times = sn1_scc.times
    new_scc._trajectory = sn1_scc._trajectory
    new_scc.SCC = scc
                
    return new_scc


def Custom_binning_function(scc_object1,scc_object2,lipid_sel,startframe,endframe,stepframe,bins,statistic,filterby):
    
    filterby=np.array(filterby)
    
    #Concatenated=np.vstack([scc_RAMP_ave.SCC,scc_PL_ave.SCC])
    
    PLs = scc_object2.tails.residues.atoms.select_atoms(lipid_sel) # This selects the lipids
    RAMP= scc_object1.tails.residues.atoms.select_atoms(lipid_sel)
    #lipids = scc_object.tails.atoms.select_atoms(lipid_sel)
    keep_RAMP = np.in1d(scc_object1.tails.residues.resindices, RAMP.residues.resindices)
    keep_PLs = np.in1d(scc_object2.tails.residues.resindices, PLs.residues.resindices)
    
    #Should be fine as long as I use same timeframe for all calculations
    start, stop, step = scc_object1.u.trajectory.check_slice_indices(startframe, endframe, stepframe)
    frames = np.arange(start, stop, step)
    keep_frames = np.in1d(scc_object1.frames, frames)
    frames = scc_object1.frames[keep_frames]
    
    scc_RAMP= scc_object1.SCC[keep_RAMP][:, keep_frames]
    scc_PLs= scc_object2.SCC[keep_PLs][:, keep_frames]
    mid_frame = frames[frames.size // 2]
    
    mid_frame_index = np.min(np.where(scc_object1.frames == mid_frame))
    filterby = filterby[keep_PLs][:, mid_frame_index]
    
    scc_object1.u.trajectory[mid_frame]
    scc_object2.u.trajectory[mid_frame]
    
    residues_PLs = PLs.groupby("resindices")
    residues_RAMP=RAMP.groupby("resindices")
    
#Potentially remove argument to unwrap?
    PLs_com = np.array([residues_PLs[res].center_of_mass(unwrap=False) for res in residues_PLs])
    RAMP_com = np.array([residues_RAMP[res].center_of_mass(unwrap=False) for res in residues_RAMP])
    
    for dim in range(3):
        PLs_com[:, dim][PLs_com[:, dim] > scc_object2.u.dimensions[dim]] -= scc_object2.u.dimensions[dim]
        PLs_com[:, dim][PLs_com[:, dim] < 0.0] += scc_object2.u.dimensions[dim]
        
        RAMP_com[:, dim][RAMP_com[:, dim] > scc_object1.u.dimensions[dim]] -= scc_object1.u.dimensions[dim]
        RAMP_com[:, dim][RAMP_com[:, dim] < 0.0] += scc_object1.u.dimensions[dim]
        
    PLs_xpos, PLs_ypos, _ = PLs_com.T 
    RAMP_xpos, RAMP_ypos, _ = RAMP_com.T 
    
    values_PLs = np.mean(scc_PLs, axis=1)
    values_RAMP = np.mean(scc_RAMP,axis=1)
    
    print(np.shape(PLs_xpos))
    
    lipids_xpos=np.hstack([PLs_xpos[filterby],RAMP_xpos])
    lipids_ypos=np.hstack([PLs_ypos[filterby],RAMP_ypos])
    values=np.hstack([values_PLs[filterby],values_RAMP])

    
    stat, x_edges, y_edges, _ = scipy.stats.binned_statistic_2d(
            x=lipids_xpos,
            y=lipids_ypos,
            values=values,
            bins=bins,statistic=statistic
        )
    
    statistic_nbins_x, statistic_nbins_y = stat.shape
    
    stat_tiled = np.tile(stat,reps=(3,3)) # assume that we do tile - default for interpolate fxn, accounts for PBC
    x,y=np.indices(stat_tiled.shape) 
    stat_tiled[np.isnan(stat_tiled)]=scipy.interpolate.griddata(
    (x[~np.isnan(stat_tiled)], y[~np.isnan(stat_tiled)]),  # points we know
           stat_tiled[~np.isnan(stat_tiled)],                     # values we know
           (x[np.isnan(stat_tiled)], y[np.isnan(stat_tiled)]),    # points to interpolate
           method="linear", # other options available, see interpolate fxn
       )
    
    stat_tiled = stat_tiled[statistic_nbins_x:statistic_nbins_x * 2, statistic_nbins_y:statistic_nbins_y * 2]    
    
    x,y=np.indices(stat.shape) 
    stat[np.isnan(stat)]=scipy.interpolate.griddata(
       (x[~np.isnan(stat)], y[~np.isnan(stat)]),  # points we know
       stat[~np.isnan(stat)],                     # values we know
        (x[np.isnan(stat)], y[np.isnan(stat)]),    # points to interpolate
        method="linear", # other options available, see interpolate fxn
       )
    
    stat = stat[statistic_nbins_x:statistic_nbins_x * 2, statistic_nbins_y:statistic_nbins_y * 2]    
    
    return stat, stat_tiled,x_edges, y_edges

def pickle_files(filename,variable):
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()
    

class Order_parameter():
    def __init__(self,tpr,traj,saving_folder,Upper_PMB1_sel, Lower_PMB1_sel,phosphates="name PO1 PO2 PO4",PL_sel="resname POPE POPG",RAMP_sel="resname RAMP",Protein_sel=None,start=0,stop=-1,step=1,verbose=True,no_bins=200,stat="mean"):
        self.tpr=tpr
        self.traj=traj
        self.phosphates=phosphates
        self.Upper_PMB1_sel=Upper_PMB1_sel
        self.Lower_PMB1_sel=Lower_PMB1_sel
        self.PL_sel=PL_sel
        self.RAMP_sel=RAMP_sel
        self.Protein_sel=Protein_sel
        self.start=start
        self.stop=stop
        self.step=step
        self.verbose=verbose
        self.universe=mda.Universe(tpr,traj)
        self.no_bins=no_bins
        self.stat=stat
        self.savingfolder=saving_folder
        
    def order_parameter_calc(self):
        leaflets=AssignLeaflets(universe=self.universe,lipid_sel=self.phosphates) # atom selection covering all lipids in the bilayer
        leaflets.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)

        
        phospholipid_leaflets=leaflets.filter_leaflets(lipid_sel=self.PL_sel)
        RAMP_leaflets=leaflets.filter_leaflets(lipid_sel=self.RAMP_sel)
        
        phospholipid_leaflet_mask= (phospholipid_leaflets == 1)
        
        L = LeafletFinder(self.universe, self.phosphates,pbc=True)
        leaflet0 = L.groups(0)
        leaflet1 = L.groups(1)
        
        POPE=leaflet0.select_atoms("resname POPE and name PO4")
        POPG=leaflet0.select_atoms('resname POPG and name PO4')
        RAMP=leaflet0.select_atoms("resname RAMP and name PO1 PO2")
        Time_positions=Calculate_Time_positions(u=self.universe,Upper_leaflet=leaflet0,Upper_PMB1s_sel=self.Upper_PMB1_sel,Lower_PMB1s_sel=self.Lower_PMB1_sel,Protein_sel=self.Protein_sel)
        Time_positions.run(verbose=self.verbose,start=self.start,stop=self.stop,step=self.step)
        print(np.shape(Time_positions.results["timeseries"]))
        LPS_phosphates_coords=Time_positions.results["timeseries"][:,0]
        POPE_phosphates_coords=Time_positions.results["timeseries"][:,1]
        POPG_phosphates_coords=Time_positions.results["timeseries"][:,2]
        
        if (self.Upper_PMB1_sel!=None) and (self.Lower_PMB1_sel!=None):
            Upper_PMB1_coords=Time_positions.results["timeseries"][:,3]
            Lower_PMB1_coords=Time_positions.results["timeseries"][:,4]
        else:
            Upper_PMB1_coords=None
            Lower_PMB1_coords=None
        
        if self.Protein_sel != None:
            Protein_coords=Time_positions.results["timeseries"][:,5]
        else:
            Protein_coords=None
        scc_PL_sn1 = run_SCC_calc(universe=self.universe,tail_sel="resname POPE POPG and name ??A",start=self.start,stop=self.stop,step=self.step)   # selects C1A, C2A, D2A, C3A, and C4A
        
        scc_PL_sn2=run_SCC_calc(universe=self.universe,tail_sel="resname POPE POPG and name ??B",start=self.start,stop=self.stop,step=self.step) 

        scc_RAMP_sn1 = run_SCC_calc(universe=self.universe,tail_sel="resname RAMP and name ??A",start=self.start,stop=self.stop,step=self.step)   # selects C1A, C2A, D2A, C3A, and C4A
        scc_RAMP_sn2 = run_SCC_calc(universe=self.universe,tail_sel="resname RAMP and name ??B",start=self.start,stop=self.stop,step=self.step)   # selects C1A, C2A, D2A, C3A, and C4A
        scc_RAMP_sn3 = run_SCC_calc(universe=self.universe,tail_sel="resname RAMP and name ??C",start=self.start,stop=self.stop,step=self.step)   # selects C1A, C2A, D2A, C3A, and C4A
        scc_RAMP_sn4 = run_SCC_calc(universe=self.universe,tail_sel="resname RAMP and name ??D",start=self.start,stop=self.stop,step=self.step)   # selects C1A, C2A, D2A, C3A, and C4A
        scc_RAMP_sn5 = run_SCC_calc(universe=self.universe,tail_sel="resname RAMP and name ??E",start=self.start,stop=self.stop,step=self.step)   # selects C1A, C2A, D2A, C3A, and C4A
        scc_RAMP_sn6 = run_SCC_calc(universe=self.universe,tail_sel="resname RAMP and name ??F",start=self.start,stop=self.stop,step=self.step)   # selects C1A, C2A, D2A, C3A, and C4A
        
        scc_RAMP_ave=weighted_average_6(sn1_scc=scc_RAMP_sn1,sn2_scc=scc_RAMP_sn2,sn3_scc=scc_RAMP_sn3,sn4_scc=scc_RAMP_sn4,sn5_scc=scc_RAMP_sn5,sn6_scc=scc_RAMP_sn6)
        scc_PL_ave=SCC.weighted_average(sn1_scc=scc_PL_sn1,sn2_scc=scc_PL_sn2)
        
        Concatenated=np.vstack([scc_RAMP_ave.SCC,scc_PL_ave.SCC])
        
        Total_results=Custom_binning_function(scc_object1=scc_RAMP_ave,scc_object2=scc_PL_ave,lipid_sel="resname RAMP POPE POPG",startframe=self.start,endframe=self.stop,stepframe=self.step,bins=self.no_bins,statistic=self.stat,filterby=phospholipid_leaflet_mask)
        
        pickle_files(f"{self.savingfolder}LPS_phosphate_coords",LPS_phosphates_coords)
        pickle_files(f"{self.savingfolder}POPE_phosphate_coords",POPE_phosphates_coords)
        pickle_files(f"{self.savingfolder}POPG_phosphate_coords",POPG_phosphates_coords)
        pickle_files(f"{self.savingfolder}Upper_PMB1_coords",Upper_PMB1_coords)
        pickle_files(f"{self.savingfolder}Lower_PMB1_coords",Lower_PMB1_coords)
        pickle_files(f"{self.savingfolder}Order_parameter",Total_results)
        pickle_files(f"{self.savingfolder}Protein_coords",Protein_coords)
        
        return LPS_phosphates_coords,POPE_phosphates_coords,POPG_phosphates_coords,Upper_PMB1_coords,Lower_PMB1_coords,Total_results, Protein_coords
        