import argparse
import math
import json
import numpy as np
import pandas as pd
from scipy import stats

import io_tools
import folded_normal_mixture

#candidate distributions which are closely related to a stable distribution
candidate_distributions_stable = {
    #"stable": stats.levy_stable, 
    "levy_leftskewed": stats.levy_l, 
    "levy": stats.levy, 
    "folded_normal": stats.foldnorm, 
    "folded_cauchy": stats.foldnorm, 
    "half_generic_normal": stats.halfgennorm, 
}

#candidate distributions which are closely related to a gamma distribution
candidate_distributions_gamma = {
    "gamma": stats.gamma, 
    "generic_gamma": stats.gengamma,
    "inverse_gamma": stats.invgamma
}

def histogram(x, nbins):
    """ Produces a PDF histogram.

        Args:
            x: data in a numpy array
            nbins: the number of bins to use

        Returns:
            A tuple of arrays (x, y), where y[i] is the estimated probability
            of observing the bin centered on x[i]. Both x and y have length nbins
    """
    y, x_edges = np.histogram(x, bins=nbins, density=True)

    bin_width = x_edges[1] - x_edges[0]
    x_ticks = (x_edges + bin_width / 2)[:-1]
    return x_ticks, y

def fit_distributions(data, candidates, nbins=50):
    """ Given a dataset and a dictionary of candidate distributions implementing
        the interface specified in scipy.stats, fits each candidate distribution 
        to the dataset and computes various values to evaluate the goodness of fit.

        Args:
            data: the dataset as a numpy array
            candidates: a dictionary which stores distributions from the 
            scipy.stats library, keyed by the name of each distribution (a string)

        Returns: 
            A dictionary with the same keys as candidates. Each value is another 
            dictionary which at least stores the parameters of the best-fit 
            distribution, and may store a number of values which can be used
            to compare goodness of fit.
    """
    fit_results = {}

    x, y = histogram(data, nbins)

    for name, dist in candidates.items():
        
        print(f"Fitting {name}")

        params = dist.fit(data)
        loc_param, scale_param = params[-2:]
        shape_params = params[:-2]

        y_hat = dist.pdf(x, loc=loc_param, scale=scale_param, *shape_params)

        sse = np.sum((y - y_hat) ** 2)

        fit_results[name] = {"params": params, "sse": sse, "sse_nbins": nbins}

    return fit_results

def fit_folded_normal_mixture(data, nbins=50):
    """ As for fit_distributions, but for the custom implementation of a mixture
        of two folded normal distributions.

        Args:            
            data: the dataset as a numpy array

        Returns:
            A dictionary {"params": ps, "sse": s, "sse_nbins": n} where ps is 
            the list of parameters for the fitted distribution, s is the sum of
            squared errors relative to a histogram of the data, and n is the 
            number of bins in the histogram
    """

    x, y = histogram(data, nbins)

    print("Fitting folded normal mixture")

    p, m1, s1, m2, s2 = folded_normal_mixture.fit(data, nbins=nbins)
    y_hat = folded_normal_mixture.mix_pdf(x, p, m1, s1, m2, s2)
    sse = np.sum((y - y_hat) ** 2)

    return {"params": [p, m1, s1, m2, s2], "sse": sse, "sse_nbins": nbins}

def get_samples_over_time(df, col_name, time_increment, max_time=None, time_col=io_tools.TIMESTEP):
    """ Given displacement data, yields the data of the chosen column at each time

        Args:
            df: a pandas DataFrame storing displacement data
            col_name: the name of the column to get data from
            time_incremement: the step size of time
            max_time: the maximum time
            time_col: the name of the time column

        Yields:
            numpy arrays of the data in df[col_name] for each time
    """

    t = time_increment
    while True:
        
        if max_time is not None and t > max_time:
            break
        
        sample_at_t = df[df[time_col] == t][col_name].dropna()
        if sample_at_t.empty:
            break

        yield t, sample_at_t.to_numpy()
        
        t += time_increment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program fits stochastic models to track data."
    )

    parser.add_argument("outfile", type=str, help="Path to the output file.")
    parser.add_argument("base_timestep", type=int, help="The smallest timestep to use.")

    parser.add_argument("--disp_file", 
        type=str, 
        help="The path to the track displacement file", 
        default=io_tools.DISPLACEMENTS_FILE_PATH
    )
    parser.add_argument("--max_timestep", type=int, help="The maximum timestep to fit distributions to.", default=None)
    parser.add_argument("--simulated", action="store_true", help="Set if the data being processed is simulated.")
    
    args = parser.parse_args()

    

    df = pd.read_csv(args.disp_file, dtype=io_tools.DTYPE_DICT)
    
    if args.simulated:
        df = io_tools.filter_simulated_displacement_data(df, remove_zeros=False)
    else:
        df = io_tools.filter_displacement_data(df, remove_zeros=False, base_timestep=args.base_timestep)
    
  
    disp_sq = df[io_tools.DISP] ** 2
    disp_sq.name = io_tools.DISP_SQ
    df = df.join(disp_sq)

    results = dict()

    for t, sample in get_samples_over_time(df, io_tools.DISP, args.base_timestep, max_time=args.max_timestep):
        
        print(f"time: {t}")

        results[t] = fit_distributions(sample, candidate_distributions_stable)
        results[t].update(fit_distributions(sample, candidate_distributions_gamma))

        folded_normal_results = fit_folded_normal_mixture(sample)
        results[t].update({"folded_normal_mixture": folded_normal_results})


    with open(args.outfile, "w") as f:
        json.dump(results, f, indent=4)
