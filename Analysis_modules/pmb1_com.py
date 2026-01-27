import numpy as np
import MDAnalysis as mda
from lipyphilic.transformations import nojump
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)
import MDAnalysis.transformations as trans
from MDAnalysis.transformations.nojump import NoJump
import pickle

# Helper functions
def COM_sel(group_a, subdivision,**kwargs):
    x_total=[]
    
    x=group_a.center_of_mass(compound=f"{subdivision}")
    
    if subdivision=="fragments":
        frag_no=group_a.fragments
    else:
        frag_no=group_a.residues
        
    mask=kwargs.get('mask', None)
    x_pre=x[mask]
    x_total.append(x_pre)
    return x_total


def COM_phosphates(Upper_phosphates,Lower_phosphates):
    Upper_positions=[]
    Lower_positions=[]
    
    Upper_positions.append(Upper_phosphates.center_of_mass(compound='group'))
    Lower_positions.append(Lower_phosphates.center_of_mass(compound='group'))
    
    return Upper_positions,Lower_positions



class COM(AnalysisBase):
    def __init__(self,tpr,traj,sel,phosphates="name PO1 PO2 PO4",xy=False,z=True,upper=True,phos_dat=False,subdivision='fragments',mask_frame=499):
        universe=mda.Universe(tpr,traj)
        universe.trajectory.add_transformations(NoJump())
        super(COM,self).__init__(universe.trajectory)
        self.tpr=tpr
        self.traj=traj
        self.universe=universe
        self.phosphates=phosphates
        self.sel=sel
        self.subdivision=subdivision
        self.xy=xy
        self.z=z
        self.upper=upper
        # Whether to save phosphate COM positions or not
        self.phos_dat=phos_dat
        self.subdivision=subdivision
        self.mask_frame=mask_frame

        L=LeafletFinder(self.universe,self.phosphates,pbc="True")
        leaflet0=L.groups(0)
        leaflet1=L.groups(1)

        self.Selection=self.universe.select_atoms(self.sel)

        self.Upper_leaflet_phosphates=leaflet0.atoms
        self.Lower_leaflet_phosphates=leaflet1.atoms

        if self.z==True:
        
            for ts in self.universe.trajectory[self.mask_frame:self.mask_frame+1]:
                COM_PMB1s_ref=self.Selection.center_of_mass(compound=self.subdivision)
                Upper_leaflet_positions=self.Upper_leaflet_phosphates.center_of_mass()
                Lower_leaflet_positions=self.Lower_leaflet_phosphates.center_of_mass()
                Middle_z=(Upper_leaflet_positions[2]+Lower_leaflet_positions[2])/2
                Upper_limit=self.universe.dimensions[2]+5
                print(Upper_limit)                
            if self.upper==True:
                Mask=(COM_PMB1s_ref[:,2]>Middle_z) & (COM_PMB1s_ref[:,2]<Upper_limit) 
                #Mask=(COM_PMB1s_ref[:,2]>0) & (COM_PMB1s_ref[:,2]<150)
            elif self.upper==False:
                Mask=(COM_PMB1s_ref[:,2]<Middle_z) | (COM_PMB1s_ref[:,2]>Upper_limit)
                #Mask=(COM_PMB1s_ref[:,2] < 0) | (COM_PMB1s_ref[:,2] > 150)
            
            self.mask=Mask

    def _prepare(self):
        self.Selection_COMs=[]
        self.Upper_phosphates=[]
        self.Lower_phosphates=[]
       

    def _single_frame(self):
        

        if self.z==True:

            Selection_COM=COM_sel(group_a=self.Selection,subdivision=self.subdivision,mask=self.mask)
                
            PMB1_no=sum((x==True) for x in self.mask)
        
            if self.xy==True:
                Total_selection=[np.array(Selection_COM)[0,x,:] for x in range(PMB1_no)]
                self.Selection_COMs.append(Total_selection)

        
            elif self.xy==False:
                Total_selection=[np.array(Selection_COM)[0,x,2] for x in range(PMB1_no)]
                self.Selection_COMs.append(Total_selection)
               
        else:
            Selection_COM=COM_sel(group_a=self.Selection,subdivision=self.subdivision)  
            if self.subdivision=="fragments":
                frag_no=len(self.Selection.fragments)
            else:
                frag_no=len(self.Selection.residues)
            #print(Selection_COM)
            Total_selection = [np.array(Selection_COM)[0,0,x,:] for x in range(frag_no)]
            self.Selection_COMs.append(Total_selection)

        if self.phos_dat!=None:
            Upper,Lower=COM_phosphates(self.Upper_leaflet_phosphates,self.Lower_leaflet_phosphates)
            self.Upper_phosphates.append(Upper)
            self.Lower_phosphates.append(Lower)
                
