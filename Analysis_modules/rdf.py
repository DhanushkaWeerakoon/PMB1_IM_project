import MDAnalysis as mda
from MDAnalysis.analysis import rdf 
import numpy
from MDAnalysis.analysis.leaflet import LeafletFinder
import pickle

# Average radial distribution functions
#Radial distribution function (RDF) describes time-averaged density of particles in b from reference group a at distance r (g_{ab}(r)). Normalised so that it is 1 for large r in a homogeneous system.

#Integrating the RDF over the spherical volume gives the radial cumilative distribution function (G_{ab}(r)).

#The average number of b particles within radius r of a at a given density is the product of the density and the radial cumilative distribution function. This can be used to compute coordination numbers e.g. number of neighbours in first solvation shell.

#The code below calculates the average radial cumilative distribution function between each atom of group b to each atom of group aover trajectory.

def pickle_files(filename,variable):
    outfile=open(filename,'wb')
    pickle.dump(variable,outfile)
    outfile.close()

class RDF():
    def __init__(self,tpr,traj,saving_folder,selA,selB,bins,binningrange,selname,verbose=True,start=0,stop=-1,step=1):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj)
        self.savingfolder=saving_folder
        self.selA=selA
        self.selB=selB
        self.bins=bins
        self.binningrange=binningrange
        self.selname=selname
        self.verbose=verbose
        self.start=start
        self.stop=stop
        self.step=step
        
    def RDF_calculator(self):
        ## About intermolecular pair distribution function (InterRDF)

        #RDF is calculated by histogramming distances between all particles in g1 and g2 while taking periodic
        #boundary conditions into account via minimum image convention. Has the following arguments:

        #- First atom group
        #- Second atom group
        #- Number of distance bins in histogram
        #- Size of RDF - range keyword specifies spherical shell around each atom (not around COM of entire group)
        #that RDF is limited to.
        
        A=self.universe.select_atoms(self.selA)
        B=self.universe.select_atoms(self.selB)
        
        irdf=rdf.InterRDF(A,B,nbins=self.bins,range=self.binningrange)
        irdf.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)
        
        pickle_files(f"{self.savingfolder}{self.selname}",irdf.results)
        
        return irdf.results
        
        
        
        
        