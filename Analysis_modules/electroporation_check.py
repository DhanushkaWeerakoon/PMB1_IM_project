import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis.base import (AnalysisBase,AnalysisFromFunction,analysis_class)


class Electroporation_check(AnalysisBase):
    def __init__(self,tpr,traj,midpoint):
        u=mda.Universe(tpr,traj,continuous=True)
        super(Electroporation_check,self).__init__(u.trajectory)
        self.tpr=tpr
        self.traj=traj
        self.universe=u
        self.midpoint=midpoint

    def _prepare(self):
        self.Upper_bilayer=self.universe.select_atoms(f"name PO1 PO2 PO4 and (prop z > {self.midpoint})")
        self.Lower_bilayer=self.universe.select_atoms(f"name PO1 PO2 PO4 and (prop z < {self.midpoint})")
        print(len(self.Upper_bilayer))
        print(len(self.Lower_bilayer))

        self.Central_NAs=[]
        self.Central_CLs=[]
        self.Central_CAs=[]
        self.Outer_NAs=[]
        self.Outer_CLs=[]
        self.Outer_CAs=[]
        self.Bilayer_COMs=[]
        

    def _single_frame(self):
        u_COM_z=self.Upper_bilayer.center_of_mass(compound='group')[2]
        l_COM_z=self.Lower_bilayer.center_of_mass(compound='group')[2]

        Bilayer_separation=u_COM_z-l_COM_z
        self.Bilayer_COMs.append(Bilayer_separation)

        Central_NA=self.universe.select_atoms(f"name NA and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
        Central_CL=self.universe.select_atoms(f"name CL and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")
        Central_CA=self.universe.select_atoms(f"name CA and (prop z < {u_COM_z}) and (prop z > {l_COM_z})")


        Outer_NA=self.universe.select_atoms(f"name NA and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
        Outer_CL=self.universe.select_atoms(f"name CL and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")
        Outer_CA=self.universe.select_atoms(f"name CA and ((prop z > {u_COM_z}) or (prop z < {l_COM_z}))")

        self.Central_NAs.append(len(Central_NA))
        self.Central_CLs.append(len(Central_CL))
        self.Central_CAs.append(len(Central_CA))

        self.Outer_NAs.append(len(Outer_NA))
        self.Outer_CLs.append(len(Outer_CL))
        self.Outer_CAs.append(len(Outer_CA))



