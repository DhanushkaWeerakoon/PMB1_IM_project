"""
Module for calculating intermolecular hydrogen bonds between two entities. Uses MDAnalysis HydrogenBondsAnalysis class.
"""
list1=[[3,4],[5,6],[7,8]]
print(np.shape(np.array(list1)))

import pickle
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
    
    
def fit_exponential(tau_timeseries: list or np.ndarray, 
                    ac_timeseries: list or np.ndarray,
                    intermittent: int =0,
                    parameters: list or None =None) -> nd.array, nd.array, nd.array, nd.array:
    """
    Fit exponential decay model to the hydrogen bond time autocorrelation function.
        
    Adapted from MDAnalysis documentation: https://userguide.mdanalysis.org/stable/examples/analysis/hydrogen_bonds/hbonds-lifetimes.html
    
    Parameters
    ----------
    tau_timeseries : array-like
        Time lag values for the autocorrelation function.
    ac_timeseries : array-like
        Autocorrelation values corresponding to tau_timeseries.
    intermittent : int, default = 0, optional
        Maximum number of frames for which a hydrogen bond is allowed to break while being considered continuous.
        If intermittent = 0, data will be fitted to a double exponential: A*exp(-t/tau1) + B*exp(-t/tau2).
        If intermittent > 0, data will be fitted to a triple exponential: A*exp(-t/tau1) + B*exp(-t/tau2) + C*exp(-t/tau3).
    parameters : list, default = None, optional
        Initial guess parameters for curve fitting. Recommended for better convergence. Should be in the form:
        For double exponential: [A, tau1, B, tau2].
        For triple exponential: [A, tau1, B, tau2, C, tau3].
        If parameters is set to None, initial parameters will be [1,1,1,1] for double exponential and [1,1,1,1,1,1] for triple exponential.
        
    Returns
    -------
    params : array
        Optimized parameters for the exponential model.
    param_covariance : array
        Covariance of the optimized parameters.
    fit_t : array
        Time values for the fitted curve.
    fit_ac : array
        Fitted autocorrelation values.
    """
    
    from scipy.optimize import curve_fit

    # Validation input parameters

    if len(tau_timeseries) != len(ac_timeseries):
        raise ValueError("tau_timeseries and ac_timeseries must have same length")
    
    if len(tau_timeseries) < 4:
        raise ValueError("Need at least 4 data points for fitting")
    
    if (type(intermittent) != int) or (intermittent < 0):
        raise ValueError("Intermittent parameter must be a non-negative integer.")  
    
    # Run curve fitting based on intermittent parameter

    if intermittent == 0:
        def model (t, A, tau1, B, tau2):
            return np.array((A*np.exp(-t/tau1) + B*np.exp(-t/tau2)))
        
        if parameters is None:
            params,param_covariance=curve_fit(model, tau_timeseries, ac_timeseries)

        else:
            if len(parameters) !=4:
                raise ValueError("For intermittent=0, parameters should be None or a list of 4 initial guess values: [A, tau1, B, tau2].")
        
            else:
                params,param_covariance=curve_fit(model, tau_timeseries, ac_timeseries, parameters)
        
    else:
        def model(t, A, tau1, B, tau2, C, tau3):
            return np.array((A*np.exp(-t/tau1)+B*np.exp(-t/tau2)+C*np.exp(-t/tau3)))
        
        if parameters is None:
            params,param_covariance=curve_fit(model, tau_timeseries, ac_timeseries)

        else:
            if len(parameters) !=6:
                raise ValueError("For intermittent>0, parameters should be None or a list of 6 initial guess values: [A, tau1, B, tau2, C, tau3].")
        
            else:
                params,param_covariance=curve_fit(model, tau_timeseries, ac_timeseries, parameters)
        
    fit_t=np.linspace(tau_timeseries[0],tau_timeseries[-1],len(tau_timeseries))
    fit_ac=model(fit_t,*params)
    
    return params, param_covariance, fit_t, fit_ac
    
    
class Hbonds_calculation():
    """    
    This class wraps MDAnalysis HydrogenBondAnalysis to compute intermolecular hydrogen bonds
    and their lifetimes from simulation trajectories.
    
    Parameters
    ----------
    tpr : str
        Path to topology file (e.g., .tpr, .gro, .pdb).
    traj : str
        Path to trajectory file (e.g., .xtc, .trr).
    hydrogens_guess : str or None
        Selection string for guessing hydrogen atoms (attached to hydrogen bond donors). 
        If None, must provide hydrogens parameter.
    acceptors_guess : str or None
        Selection string for guessing acceptor atoms. If None, must provide
        acceptors parameter.
    between : list, default = None, optional
        Specify two selection strings for non-updating atom groups between which 
        hydrogen bonds will be calculated.
        List can be one-dimensional e.g. ["protein","resname SOL"], which will calculate
        protein-water hydrogen bonds.
        List can be two dimensional e.g. [["protein", "resname SOL"],["protein","protein"]],
        which will calculate both protein-water and protein-protein hydrogen bonds.
    update : bool, default=False, optional
        Whether to update atom selections each frame.
    distance_cutoff : float, default=3.5, optional
        Maximum donor-acceptor distance (Angstroms).
    angle_cutoff : float, default=150, optional
        Minimum donor-hydrogen-acceptor angle (degrees).
    acceptor_charge : float, default=-0.5, optional
        Maximum partial charge for acceptor guess.
    donors : str, default=None, optional
        Explicit hydrogen bond donor atom selection string.
    hydrogens : str, default=None, optional
        Explicit hydrogen (attached to hydrogen bond donor) atom selection string.
    acceptors : str, default=None, optional
        Explicit acceptor atom selection string.
    verbose : bool, default=True, optional
        Print progress information.
    start : int, default=0, optional
        First frame to analyze.
    stop : int, default=-1, optional
        Last frame to analyze (-1 for all).
    step : int, default=1, optional
        Analyze every nth frame.
        
    Attributes
    ----------
    All parameters.
    universe : MDAnalysis.Universe
        The loaded trajectory universe.
    hbonds_results : HydrogenBondAnalysis
        Analysis object containing H-bond data (after calling calculation()).
    times : array
        Simulation times (ps) per frame over which hydrogen bonds are calculated (after 
        calling calculation()).
    hbonds_timeseries : array
        Number of intermolecular hydrogen bonds per frame (after calling calculation()).
    hbonds_type : dict
        Hydrogen bond count totals (by donor-acceptor pair types) summed over all frames
        (after calling calculation()).
    """
    
    
    def __init__(self,tpr: str, traj: str,hydrogens_guess: str or None,
                 acceptors_guess: str or None, 
                 between: list or None = None,
                 update: bool = False, distance_cutoff: float = 3.5,
                 angle_cutoff: float = 150,
                 acceptor_charge: float =-0.5,
                 donors: str or None = None, hydrogens: str or None = None,
                 acceptors: str or None = None,verbose: bool = True,
                 start: int=0, stop: int=-1, step: int=1):
        self.tpr=tpr
        self.traj=traj
        self.universe=mda.Universe(tpr,traj)
        self.update=update
        self.verbose=verbose
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

    
        if (type(self.start) != int) or (self.start < 0):
            raise ValueError("start parameter must be a non-negative integer.")
        
        if (type(self.stop) != int) or (self.stop < -1):
            raise ValueError("stop parameter must be an integer greater than or equal to -1.")
        
        if (type(self.step) != int) or (self.step < 1):
            raise ValueError("step parameter must be a positive integer.")

        if between is not None:
            between_shape=np.shape(np.array(self.between))

            if len (between_shape) > 2:
                raise ValueError("between parameter must be a one- or two-dimensional list.") 
            
            if len(between_shape) == 2:
                if between_shape[1] != 2:
                    raise ValueError("between parameter must be a two-dimensional list, with each constituent list containing two values.")

            if len(between_shape) == 1:
                if len(self.between) != 2:
                    raise ValueError("between parameter must be a one-dimensional list containing two values.")


            if len (between_shape) == 0:
                raise ValueError("between parameter cannot be an empty list.")   


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
        
    # Fix this next - run in Jupyter as well to see if it works.
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
        

