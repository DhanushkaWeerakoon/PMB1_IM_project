"""
Module for calculating intermolecular hydrogen bonds and their lifetimes using MDAnalysis.
"""



import pickle
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
    
    
def fit_exponential(tau_timeseries, ac_timeseries,intermittent,parameters):
    """
    Fit exponential decay model to hydrogen bond autocorrelation data.
    
    Adapted from MDAnalysis tutorials.
    
    Parameters
    ----------
    tau_timeseries : array-like
        Time lag values for the autocorrelation function.
    ac_timeseries : array-like
        Autocorrelation values corresponding to tau_timeseries.
    intermittent : int
        Maximum number of frames for which a hydrogen bond is allowed to break.
    parameters : array-like
        Initial guess parameters for curve fitting.
        
    Returns
    -------
    params : array
        Optimized parameters for the exponential model.
    fit_t : array
        Time values for the fitted curve.
    fit_ac : array
        Fitted autocorrelation values.
        
    Notes
    -----
    - Double exponential: A*exp(-t/tau1) + B*exp(-t/tau2) - used when intermittent = 0,
    - Triple exponential: A*exp(-t/tau1) + B*exp(-t/tau2) + C*exp(-t/tau3) - used when intermittent > 0.
    """
    
    from scipy.optimize import curve_fit
    
    if intermittent == 0:
        def model (t, A, tau1, B, tau2):
            return (A*np.exp(-t/tau1) + B*np.exp(-t/tau2))
        params, param_covariance=curve_fit(model,tau_timeseries, ac_timeseries,parameters)
        
    else:
        def model(t, A, tau1, B, tau2, C, tau3):
            return (A*np.exp(-t/tau1)+B*np.exp(-t/tau2)+C*np.exp(-t/tau3))
        params,param_covariance=curve_fit(model, tau_timeseries, ac_timeseries, parameters)
        
    fit_t=np.linspace(tau_timeseries[0],tau_timeseries[-1],len(tau_timeseries))
    fit_ac=model(fit_t,*params)
    
    return params, fit_t, fit_ac
    
    
class Hbonds_calculation():
    """    
    This class wraps MDAnalysis HydrogenBondAnalysis to compute intermolecular hydrogen bonds
    and their lifetimes from simulation trajectories.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .pdb).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    hydrogens_guess : str or None
        Selection string for guessing hydrogen atoms (attached to hydrogen bond donors). 
        If None, must provide hydrogens parameter.
    acceptors_guess : str or None
        Selection string for guessing acceptor atoms. If None, must provide
        acceptors parameter.
    between : list of lists of string pairs, optional
        Each item in the list contains two atom selection strings restricting the
        hydrogen bond search between groups.
    update : bool, default=False
        Whether to update atom selections each frame.
    distance_cutoff : float, default=3.5
        Maximum donor-acceptor distance (Angstroms).
    angle_cutoff : float, default=150
        Minimum donor-hydrogen-acceptor angle (degrees).
    acceptor_charge : float, default=-0.5
        Maximum partial charge for acceptor guess.
    donors : str, optional
        Explicit hydrogen bond donor atom selection string.
    hydrogens : str, optional
        Explicit hydrogen (attached to hydrogen bond donor) atom selection string.
    acceptors : str, optional
        Explicit acceptor atom selection string.
    verbose : bool, default=True
        Print progress information.
    start : int, default=0
        First frame to analyze.
    stop : int, default=-1
        Last frame to analyze (-1 for all).
    step : int, default=1
        Analyze every nth frame.
        
    Attributes
    ----------
    All parameters.
    universe : MDAnalysis.Universe
        The loaded trajectory universe.
    hbonds_results : HydrogenBondAnalysis
        Analysis object containing H-bond data (after calling calculation()).
    times : array
        Simulation time frames (ps) over which hydrogen bonds are calculated (after calling
        calculation()).
    hbonds_timeseries : array
        Number of intermolecular hydrogen bonds per frame (after calling calculation()).
    hbonds_type : dict
        Hydrogen bond count totals (by donor-acceptor pair types) summed over all frames
        (after calling calculation()).
    """
    
    
    def __init__(self,tpr,traj,hydrogens_guess,acceptors_guess,between=None,update=False,distance_cutoff=3.5,angle_cutoff=150,acceptor_charge=-0.5,donors=None,hydrogens=None,acceptors=None,verbose=True,start=0,stop=-1,step=1):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj)
        self.update=update
        self.verbose=True
        self.start=start
        self.stop=stop
        self.step=step
        self.between=between
        self.distance_cutoff=distance_cutoff
        self.angle_cutoff=angle_cutoff
        self.donors=donors
        self.hydrogens=hydrogens
        self.acceptors=acceptors   
        self.hydrogens_guess=hydrogens_guess
        self.acceptors_guess=acceptors_guess
        self.acceptor_charge=acceptor_charge
        
    def calculation(self):
        hbonds_container=HydrogenBondAnalysis(universe=self.universe,update_selections=self.update, between=self.between, d_a_cutoff=self.distance_cutoff, d_h_a_angle_cutoff=self.angle_cutoff, donors_sel=self.donors, hydrogens_sel=self.hydrogens,acceptors_sel=self.acceptors)
        
        if self.hydrogens_guess != None:
            Hydrogens_sel=hbonds_container.guess_hydrogens(self.hydrogens_guess)            
            hbonds_container.hydrogen_sel=f"{Hydrogens_sel}"
            
            
        if self.acceptors_guess !=None:
            Acceptors_sel=hbonds_container.guess_acceptors(self.acceptors_guess,max_charge=self.acceptor_charge)
            hbonds_container.acceptors_sel=f"{Acceptors_sel}"
            
        hbonds_container.run(verbose=self.verbose,step=self.step,start=self.start,stop=self.stop)
        self.hbonds_results=hbonds_container
        self.times=hbonds_container.times
        self.hbonds_timeseries=hbonds_container.count_by_time()
        self.hbonds_type=hbonds_container.count_by_type()
        
    def lifetime_calc(self,window,tau_max,parameters,intermittent=0):

        """
        Calculate hydrogen bond lifetimes and fit exponential decay.
        
        Parameters
        ----------
        window : int
            Window step size for lifetime calculation.
        tau_max : int
            Maximum tau value (in frames) for lifetime calculation.
        parameters : array-like
            Initial parameters for exponential fit.
        intermittent : int, default=0
            Maximum number of frames for which a hydrogen bond is allowed to break.
            
        Sets Attributes
        ---------------
        intermittent : int
            Stored intermittency parameter.
        lifetime_window : int
            Stored window parameter.
        lifetime_tau_max : int
            Stored tau_max parameter.
        parameters : array-like
            Stored initial fit parameters.
        params : array
            Fitted exponential parameters.
        fit_t : array
            Time values for fitted curve.
        fit_ac : array
            Fitted autocorrelation values.
        tau_frames : array
            Time lag values from lifetime analysis.
        hbond_lifetimes : array
            Autocorrelation values from lifetime analysis.
            
        Notes
        -----
        Must call calculation() before calling this method.
        """
        
        self.intermittent=intermittent
        self.lifetime_window=window
        self.lifetime_tau_max=tau_max
        self.parameters=parameters
        
        tau_frames, hbond_lifetimes=self.hbonds_results.lifetime(tau_max=self.lifetime_tau_max,window_step=self.lifetime_window,intermittency=self.intermittent)
        params, fit_t, fit_ac=fit_exponential(tau_frames, hbond_lifetimes, intermittent=self.intermittent,parameters=self.parameters)
        self.params=params
        self.fit_t=fit_t
        self.fit_ac=fit_ac
        self.tau_frames=tau_frames
        self.hbond_lifetimes=hbond_lifetimes
        
        
