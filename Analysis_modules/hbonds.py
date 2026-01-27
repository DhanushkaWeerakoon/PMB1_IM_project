import pickle
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
    
    
def fit_exponential(tau_timeseries, ac_timeseries,intermittent,parameters):
    """ From MDA tutorials"""
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
        
        
