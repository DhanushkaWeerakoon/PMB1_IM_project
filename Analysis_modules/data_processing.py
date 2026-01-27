import numpy as np
import pickle

# Blocked_apl was build with help from ChatGPT 

def pickle_save(filename,variable):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)
    
def pickle_open(filename):
    with open(filename, 'rb') as f:
        variable=pickle.load(f)
    return variable
        
    

def histogram_calc(function,bins,binning_range,lim):
    """
    Takes a one-dimensional function, and histograms data into specified x bins within binning range
    function: 1D array with results
    bins: number of bins (integer)
    binning_range: range of bins (lower limit (integer), upper limit (integer))
    lim: np.histogram outputs bin edges, not actual bins - discard last edg
    """
    counts,bins=np.histogram(function,bins=bins,range=binning_range)
    
    #Normalises counts
    counts_scaled=counts/sum(counts)
    #Calculate bin centres    
    bin_centres=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
  
    # Calculate expectation value of histogram
    exp_value=np.sum(bin_centres*counts_scaled)
    # Calculate standard deviation of histogram - mark down that this is the formula that you used
    std=np.sqrt(np.sum(((bin_centres-exp_value)**2)*counts_scaled))
    # Returns bins(array), scaled counts(array), expectation value(single value) and standard deviation (single value)
    return bins,counts_scaled,exp_value,std



def blocked(function,nblocks,nbins,binning_range,bootstrapped=None):
    # Takes a one-dimensional function, divides it into n blocks and histograms data in each block into specified bins (defined by nbins and binning range), with bootstraps if needed
    
    raw_blocks=[]
    
               
    size=int(len(function)/nblocks)
    # Divide up function into blocks and operate on block by block to get histogrammed values
    for block in range(nblocks):
        f=function[block*size:(block+1)*size]
        f=np.ravel(f)
        print(np.shape(f))
        raw_blocks.append(f)
        
    if bootstrapped == None:
        hist_exp_values=[]
        hist_std_values=[]
        hist_bins_values=[]
        hist_counts_values=[]
        for i in raw_blocks:
            hist_bins, hist_counts_scaled, hist_exp_value, hist_std=histogram_calc(function=i, bins=nbins, binning_range=binning_range,lim=nbins)
            hist_exp_values.append(hist_exp_value)
            hist_std_values.append(hist_std)
            hist_bins_values.append(hist_bins)
            hist_counts_values.append(hist_counts_scaled)
        print(hist_exp_values)
            
        return np.average(hist_exp_values), np.average(hist_std_values), hist_bins_values, hist_counts_values, np.std(hist_exp_values,ddof=1)/np.sqrt(len(hist_exp_values))
    
    # If bootstrapping is activated, use code below    
    if bootstrapped != None:
        bootstraps_total=[]
        exp_values_bootstrap=[]
        std_values_bootstrap=[]
        bins_bootstrap=[]
        counts_bootstrap=[]

        # Iterates for a specified number of bootstraps
        for _ in range(bootstrapped):
            # Make a list of blocks of length nblocks using items in previously generated raw_blocks list with replacement 
            bootstraps=[]
            bootstrap_indices = np.random.choice(nblocks,size=nblocks,replace=True)
            bootstraps.append([raw_blocks[i] for i in bootstrap_indices])
            
            # Histogram and calculate stats for each block within bootstrapped set of blocks
            hist_exp_values=[]
            hist_std_values=[]
            hist_counts_values=[]
            hist_bins_values=[]

            for j in bootstraps:
                hist_bins,hist_counts_scaled,hist_exp_value,hist_std=histogram_calc(function=j,bins=nbins,binning_range=binning_range,lim=nbins)
                hist_exp_values.append(hist_exp_value)
                hist_std_values.append(hist_std)
                hist_bins_values.append(hist_bins)
                hist_counts_values.append(hist_counts_scaled)
            
            # Collect all bootstrapped histogram data - average expectations values and standard deviations across blocks for each bootstrap
            exp_values_bootstrap.append(np.average(hist_exp_values))
            std_values_bootstrap.append(np.average(hist_std_values))
            bins_bootstrap.append(hist_bins_values)
            counts_bootstrap.append(hist_counts_values)  
        return exp_values_bootstrap, std_values_bootstrap, bins_bootstrap, counts_bootstrap
        
    
def test_blocking(fxn,block_no,bins,binning_range,apl=False):
    block_size=[]
    averages=[]
    stds=[]
    bins_total=[]
    counts_total=[]
    error_of_mean=[]
    
    # Iterate over a range of block sizes
    for i in range(1,block_no+1):
        # Determine if function can be divided into even blocks by block size in question - if it does, proceed with calculation.
        if (len(fxn) % i == 0):
            # Save block members information for later plotting
            block_size.append(len(fxn)/i)
            # Calculation
            
            
            exp_values,std_values,bin_values,counts_values, exp_std_values=blocked(function=fxn,nblocks=i,nbins=bins,bootstrapped=None,binning_range=binning_range)       
            bins_total.append(bin_values)
            counts_total.append(counts_values)
            averages.append(exp_values)
            stds.append(std_values)
            error_of_mean.append(exp_std_values)
    return averages, stds, bins_total,counts_total, block_size,error_of_mean




def tail_calculation(tail_length,function,block_no,bin_no,bootstraps,binning_range):
    if bootstraps!=None:
        bootstrap_aves=[]
        bootstrap_stds=[]
        bootstrap_counts=[]
        bootstrap_bins=[]
        
        for i in range(tail_length):
            bootstrap_ave,bootstrap_std,bootstrap_bin,bootstrap_count=blocked(function=function[:,i],nblocks=block_no,nbins=bin_no,bootstrapped=bootstraps,binning_range=binning_range)
            bootstraps_aves.append(np.average(bootstrap_ave))
            bootstrap_stds.append(np.average(bootstrap_std))
            bootstrap_counts.append(bootstrap_count)
            bootstrap_bins.append(bootstrap_bin)
            
        return bootstrap_aves, bootstrap_stds, bootstrap_counts, bootstrap_bins
    
    else:
        aves=[]
        stds=[]
        counts=[]
        bins=[]
        errors=[]
        
        for i in range(tail_length):
            ave,std,binned,count,error_of_mean=blocked(function=function[:,i],nblocks=block_no,nbins=bin_no,bootstrapped=None,binning_range=binning_range)
            aves.append(ave)
            stds.append(std)
            counts.append(count)
            bins.append(binned)
            errors.append(error_of_mean)
            
        return aves,stds,counts,bins,errors