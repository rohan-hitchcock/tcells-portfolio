import numpy as np
from scipy import optimize
import time
import math

#the folded normal probability density function
fn_pdf = lambda x, m, s : (np.exp((x - m) ** 2 / (-2 * abs(s))) + np.exp( (x + m) ** 2 / (-2 * abs(s)))) / np.sqrt(2 * np.pi * abs(s))

#density function of the mixture of two folded normals
mix_pdf = lambda x, p, m1, s1, m2, s2 : p * fn_pdf(x, m1, s1) + (1 - p) * fn_pdf(x, m2, s2)

def fit_folded_normal(xs, guess_m=None, guess_s=None):
    """ Finds the maximum likelyhood estimator of a folded normal distribution 
        fit to sample xs.

        Args:
            xs: the sample as a numpy array
            guess_m: an optional guess for the mean of the underlying normal distribution
            guess_s: an optional guess for the variance of the underlying normal distribution

        Returns:
            A tuple (m, s) where m is the mean and s is the variance of the 
            underlying normal distribution of the MLE for xs.
    """
    def L(params, xs):
        """ Minimizing this funciton is equivalent to maximizing log-likelyhood"""
        m, s = params
        val = -np.sum((
            -np.log(2 * np.pi * abs(s)) / 2 - ((xs - m) ** 2) / (2 * abs(s)) 
            + np.log(1 + np.exp(((-2 * m) / abs(s)) * xs))))
        return val

    s = np.std(xs)

    result = optimize.minimize(L, [(1/3) * np.mean(xs), s], args=(xs,), method="Nelder-Mead", options={"maxiter": 10000})

    if not result.success:
        print(f"From optimizer: {result.message} (status: {result.status})")

    return result.x

def estimate_params(xs, nbins, min_data_size=1):
    """ Given a sample xs, finds an inital estimate for a folded normal 
        distribution to fit this data. This result is designed to be used as 
        the seed for a more sophisticated fitting algorithm.

        Note this function is specialised to the expected distribution of the 
        T cell displacement data. 

        Args:
            nbins: A hyperparameter which controls the number of bins being 
            used in a histogram of the data
        
        Returns:
            A tuple (p, m1, s1, m2, s2) which parameterises a mixture of two 
            folded normal dsitrubution
    """
    y, x_edges = np.histogram(xs, bins=nbins, density=False)

    first_bin_idx = xs <= x_edges[1]

    xs_first_bin = xs[first_bin_idx]
    xs_tail = xs[~first_bin_idx]

    dist_1_count = y[0] - y[1]
    if dist_1_count < min_data_size - 1:
        print(f"WARNING: no peak at zero. Initial parameter estimation fails using nbins={nbins}.")
        new_nbins = math.floor(0.75 * nbins)
        print(f"trying again with less bins... (nbins={new_nbins})")
        return estimate_params(xs, new_nbins)

    rng = np.random.default_rng()
    
    rng.shuffle(xs_first_bin)


    dist_1_data = xs_first_bin[:dist_1_count]

    dist_2_data = np.append(xs_tail, xs_first_bin[dist_1_count:])
    rng.shuffle(dist_2_data)

    m1, s1 = fit_folded_normal(dist_1_data)
    m2, s2 = fit_folded_normal(dist_2_data)
    mix_p = len(dist_1_data) / (len(dist_1_data) + len(dist_2_data))

    return mix_p, m1, s1, m2, s2

def em_2_folded_normals(xs, mix_p, mu_1, sig2_1, mu_2, sig2_2, method="Nelder-Mead", conv_tol=1e-12):
    """ The Expectation-Maximisation algorithm implemented to fit a mixture of 
        two folded normal distributions. A random variable Y ~ |X|, where X is
        normally distributed, has a folded normal distribution. A folded normal
        distribution is parametised by the mean mu and variance sigma^2 of the 
        underlying normal distribution.
        
        Args:
            xs: The data to fit, in a numpy array.
            mix_p: a guess for the mixing probability.
            mu_1: a guess for the location parameter of one folded normal
            mu_2 a guess for the location parameter of the other folded normal
            sig2_1: a guess for the scale paramteter of one folded normal
            sig2_2: a guess for the scale paramteter of the other folded normal
            method: Method to use in the optimizer which maximises the log-likelyhood
            refer to scipy.optimize.minimize for options.
            conv_tol: Determines when convergence has occured. 

        Returns:
            An improved estimate of the parameters.
    """

    def F(params, xs, gs):
        """ Minimizing this function is equivalent to maximising the 
            log likelihood function. Returns a tuple (f, j) where f is the 
            value of the function and j is the value of its Jacobian.
        """
        m1, s1, m2, s2 = params

        #compute value of function
        f = -np.sum(
            (gs * ( 
                -np.log(2 * np.pi * abs(s1)) / 2 - ((xs - m1) ** 2) / (2 * abs(s1)) 
                + np.log(1 + np.exp(((-2 * m1) / abs(s1)) * xs))
            )) + ((1 - gs) * (
                -np.log(2 * np.pi * abs(s2)) / 2 - ((xs - m2) ** 2) / (2 * abs(s2)) 
                + np.log(1 + np.exp(((-2 * m2) / abs(s2)) * xs))
            ))
        )
        return f

    old_gs = None
    while True:
    
        # expectation step 

        #compute `probability` each element of xs belongs to distribution 1
        a = mix_p * fn_pdf(xs, mu_1, sig2_1)
        gs = a / (a + (1 - mix_p) * fn_pdf(xs, mu_2, sig2_2))


        #check if we have converged within the threshold, if so: exit
        if old_gs is not None and np.allclose(gs, old_gs, atol=conv_tol):
            break
        
        # maximisation step
        result = optimize.minimize(F, [mu_1, sig2_1, mu_2, sig2_2], args=(xs, gs), jac=False, method=method, options={"maxiter": 10000})

        mix_p = np.mean(gs)
        
        if not result.success:
            print(f"From optimizer: {result.message} (status: {result.status})")

        mu_1, sig2_1, mu_2, sig2_2 = result.x
        
        old_gs = gs

    return mix_p, mu_1, mu_2, sig2_1, sig2_2

def fit(xs, nbins=50, conv_tol=1e-12, minimizer="Nelder-Mead"):
    """ Given a sample xs, attempts to fit a mixture of two folded normal 
        distributions. 

        Uses the expectation-maximisation (E-M) algorithm. An initial parameter
        guess is made based on assumptions about the distribution of T cell displacement data.

        Args:
            xs: sample as a numpy array
            nbins: a hyperparameter to the initial parameter guessing procedure
            conv_tol: controls the what we consider 'convergence' in the E-M
            algorithm ('converge' once all responsibilities change by less than this)
            minimizer: the name (a string) of the minimizer to use in the E-M 
            algorithm. Refer to scipy.optimize.minimize for allowed options.

        Returns:
            A tuple (p, m1, s1, m2, s2) where p is the mixing parameter, (m1, s1)
            are the parameters of the first folded normal distribution (the 
            mean and variance of the underlying normal distribution), and 
            similarly for (m2, s2).
    """
    
    p, m1, s1, m2, s2 = estimate_params(xs, nbins)

    return em_2_folded_normals(xs, p, m1, s1, m2, s2, method=minimizer, conv_tol=conv_tol)

# Functions to demonstrate parts of this package ******************************

def test_mixture_fitting():
    import matplotlib.pyplot as plt
    from scipy import stats


    data_size = 10000

    true_m1 = np.random.uniform(1.5, 6) / 10
    true_m2 = np.random.uniform(0, 1) / 10

    true_s1 = np.random.uniform(1, 5) / 10
    true_s2 = np.random.uniform(0.1, 0.4) / 10
    
    true_mix_p = np.random.uniform(0.2, 0.8)
    
    num_1 = round(true_mix_p * data_size)
    num_2 = data_size - num_1


    data_1 = np.abs(stats.norm.rvs(loc=true_m1, scale=true_s1, size=num_1))
    data_2 = np.abs(stats.norm.rvs(loc=true_m2, scale=true_s2, size=num_2))

    data = np.append(data_1, data_2)
    np.random.shuffle(data)

    
    print(f"Size: {data_size}")
    print()
    print("True values:")
    print(f"params 1: (mu, sig^2) = ({true_m1}, {true_s1 ** 2})")
    print(f"params 2: (mu, sig^2) = ({true_m2}, {true_s2 ** 2})")
    print(f"mixing: {len(data_1) / data_size}")
    print()


    guess_m1 = abs(true_m1 + np.random.uniform(-0.1, 0.1))
    guess_m2 = abs(true_m2 + np.random.uniform(-0.1, 0.1))

    guess_s1 = abs(true_s1 ** 2 + np.random.uniform(-0.1, 0.1))
    guess_s2 = abs(true_s2 ** 2 + np.random.uniform(-0.1, 0.1))
    
    guess_mix_p = true_mix_p + np.random.uniform(-0.1, 0.1)

    print("Inital guesses:")
    print(f"params 1: (mu, sig^2) = ({guess_m1}, {guess_s1})")
    print(f"params 2: (mu, sig^2) = ({guess_m2}, {guess_s2})")
    print(f"mixing: {guess_mix_p}")
    print()


    mix_p, m1, m2, s1, s2 = em_2_folded_normals(data, guess_mix_p, guess_m1, guess_s1, guess_m2, guess_s2, conv_tol=1e-12, method="Nelder-Mead")

    print("Fitted values:")
    print(f"params 1: (mu, sig^2) = ({m1}, {s1})")
    print(f"params 2: (mu, sig^2) = ({m2}, {s2})")
    print(f"mixing: {mix_p}")

    line = np.linspace(min(data), max(data), num=1000)
    pdf_pts = mix_p * fn_pdf(line, m1, s1) + (1 - mix_p) * fn_pdf(line, m2, s2)
    true_pts = mix_pdf(line, true_mix_p, true_m1, true_s1 ** 2, true_m2, true_s2 ** 2)
    guess_pts = guess_mix_p * fn_pdf(line, guess_m1, guess_s1) + (1 - mix_p) * fn_pdf(line, guess_m2, guess_s2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(line, true_pts, c='g', alpha=0.5)
    ax.plot(line, guess_pts, c='k', alpha=0.5)
    ax.plot(line, pdf_pts, c='r')
    

    ax.hist(data, density=True, histtype='stepfilled', alpha=0.2)

    plt.show()

def test_folded_fitting():
    import matplotlib.pyplot as plt
    from scipy import stats

    data_size = 10000

    true_m = np.random.uniform(1.5, 6) / 10
    true_s = np.random.uniform(1, 5) / 10
    
    data = np.abs(stats.norm.rvs(loc=true_m, scale=true_s, size=data_size))

    print(f"Size: {data_size}")
    print()
    print("True values:")
    print(f"params: (mu, sig^2) = ({true_m}, {true_s ** 2})")
    print()

    fit_m, fit_s = fit_folded_normal(data)

    print("Fitted values:")
    print(f"params 1: (mu, sig^2) = ({fit_m}, {fit_s})")
    
    line = np.linspace(min(data), max(data), num=1000)
    pdf_pts = fn_pdf(line, fit_m, fit_s)
    true_pts = fn_pdf(line, true_m, true_s ** 2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(line, true_pts, c='g', alpha=0.5)
    ax.plot(line, pdf_pts, c='r')

    ax.hist(data, density=True, histtype='stepfilled', alpha=0.2)

    plt.show()

if __name__ == "__main__":
    test_folded_fitting()
    test_mixture_fitting()
