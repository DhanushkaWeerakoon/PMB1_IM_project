"""
Module for calculating radial distribution functions (RDF) and radial cumulative distribution functions between molecular selections. RDFs describe the time-averaged density of particles as a function of distance and are fundamental for understanding molecular structure and solvation.

Background
----------
The radial distribution function g_ab(r) describes the time-averaged density
of particles in group b at distance r from particles in reference group a.
It is normalized such that g_ab(r) → 1 for large r in a homogeneous system.

Integrating the RDF over a spherical volume gives the radial cumulative
distribution function G_ab(r), which represents the average number of b
particles within radius r of an a particle.

The average number of b particles within radius r of a at a given density is the product of the density and the radial cumilative distribution function. This can be used to compute coordination numbers e.g. number of neighbours in first solvation shell.
"""


import MDAnalysis as mda
from MDAnalysis.analysis import rdf 
import numpy
from MDAnalysis.analysis.leaflet import LeafletFinder
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

class RDF():
    """
    Calculate radial distribution functions between two selections.
    
    This class computes the intermolecular radial distribution function
    (RDF) between two atom groups over a trajectory. The RDF quantifies
    the probability of finding particles from group B at distance r from
    particles in group A, normalized by the expected density in a
    homogeneous system.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    saving_folder : str
        Directory path where RDF results will be saved.
    selA : str
        MDAnalysis selection string for reference group (group A).
    selB : str
        MDAnalysis selection string for target group (group B).
    bins : int
        Number of distance bins for the RDF histogram.
    binningrange : tuple of float
        Distance range (min, max) in Angstroms for RDF calculation.
        Example: (0.0, 15.0) for 0-15 Å range.
    selname : str
        Base name for output pickle file.
    verbose : bool, default=True
        Print progress information during analysis.
    start : int, default=0
        First frame to analyze.
    stop : int, default=-1
        Last frame to analyze (-1 for all frames).
    step : int, default=1
        Analyze every nth frame.
        
    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    savingfolder : str
        Directory for saving results.
    selname : str
        Base filename for outputs.
        
    
    Notes
    -----
    - Uses MDAnalysis.analysis.rdf.InterRDF for calculation
    - Periodic boundary conditions handled via minimum image convention
    - Distance range (binningrange) specifies spherical shell around
      EACH atom in group A, not around center of mass
    - Bins are histogram centers, not edges
    - Results averaged over all frames and all particle pairs
    
    References
    ----------
    .. [1] Allen, M. P., & Tildesley, D. J. (2017). Computer simulation
           of liquids. Oxford university press.
    """
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
        """
        Calculate the radial distribution function between selections.
        
        Computes the intermolecular pair distribution function (RDF)
        between group A and group B by histogramming distances between
        all atom pairs while accounting for periodic boundary conditions.
        
        Returns
        -------
        results : MDAnalysis.analysis.rdf.Results
            Results object containing:
            - bins : ndarray
                Array of bin centers (Å). Shape: (nbins,)
            - rdf : ndarray
                Radial distribution function g(r) values (dimensionless).
                Shape: (nbins,)
            - count : ndarray
                Radial cumulative distribution G(r), representing the
                average number of B particles within radius r of A
                particles. Shape: (nbins,)

        """

        # Select atom groups
        A=self.universe.select_atoms(self.selA)
        B=self.universe.select_atoms(self.selB)

        # Calculate intermolecular RDF
        irdf=rdf.InterRDF(A,B,nbins=self.bins,range=self.binningrange)
        irdf.run(start=self.start,stop=self.stop,step=self.step,verbose=self.verbose)

        # Save results to pickle file
        pickle_files(f"{self.savingfolder}{self.selname}",irdf.results)
        
        return irdf.results
        
        
        
        
        