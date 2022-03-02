import numpy as np
from scipy import optimize
import time

#the folded normal probability density function
fn_pdf = lambda x, m, s2 : (np.exp((x - m) ** 2 / (-2 * s2)) + np.exp( (x + m) ** 2 / (-2 * s2))) / np.sqrt(2 * np.pi * s2)


def estimate_params(xs, nbins):

    y, x_edges = np.histogram(xs, bins=nbins, density=True)

    first_bin_idx = xs <= x_edges[1]

    xs_first_bin = xs[first_bin_idx]
    xs_tail = xs[~first_bin_idx]

    dist_1_count = y[1] - y[0]
    assert dist_1_count > 0

    rng = np.random.default_rng()
    
    rng.shuffle(xs_first_bin)


    dist_1_data = xs_first_bin[:dist_1_count]

    dist_2_data = xs_tail.append(xs_first_bin[dist_1_count:])
    rng.shuffle(dist_2_data)



def fit_folded_normal(xs):


    def f(params, xs):

        m, s = params

        return -fn_pdf(xs, m, s)


    s = np.std(xs)

    result = optimize.minimize(f, [0, s], args=(xs,), method="Nelder-Mead", options={"maxiter": 10000})

    if not result.success:
        print(f"From optimizer: {result.message} (status: {result.status})")

    return result.x


def em_2_folded_normals(xs, mix_p, mu_1, mu_2, sig2_1, sig2_2, method="Nelder-Mead", conv_tol=1e-12):
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
            log likelyhood function. Returns a tuple (f, j) where f is the 
            value of the function and j is the value of its Jacobian.
        """
        m1, s1, m2, s2 = params

        #precompute some complex terms used more than once
        xs_div_exp_term_1 = np.divide(xs, 1 + np.exp(((2 * m1) / s1) * xs))
        xs_div_exp_term_2 = np.divide(xs, 1 + np.exp(((2 * m2) / s2) * xs))

        #compute value of function
        f = -np.sum(
            (gs * ( 
                -np.log(2 * np.pi * s1) / 2 - ((xs - m1) ** 2) / (2 * s1) 
                + np.log(1 + np.exp(((-2 * m1) / s1) * xs))
            )) + ((1 - gs) * (
                -np.log(2 * np.pi * s2) / 2 - ((xs - m2) ** 2) / (2 * s2) 
                + np.log(1 + np.exp(((-2 * m2) / s2) * xs))
            ))
        )

        #compute Jacobian
        j1 = -np.sum((gs / s1) * ((xs - (2 * xs_div_exp_term_1)) - m1))

        j2 = -np.sum(gs * (-1 / (2 * s1) + ((xs - m1) ** 2) / (2 * s1 ** 2) + (2 * m1 / (s1 ** 2)) * xs_div_exp_term_1))                

        j3 = -np.sum(((1 - gs)  / s2) * ((xs - (2 * xs_div_exp_term_2)) - m2))

        j4 = -np.sum((1 - gs) * (-1 / (2 * s2) + ((xs - m2) ** 2) / (2 * s2 ** 2) + (2 * m2 / (s2 ** 2)) * xs_div_exp_term_2))

        return (f, np.array([j1, j2, j3, j4]))
            
        
    def hess(params, xs, gs):
        """ Returns the Hessian of F"""
        m1, s1, m2, s2 = params
        
        #precompute some complex terms used more than once            
        xs_div_exp_term_1 = np.divide(xs, 1 + np.exp(((2 * m1) / s1) * xs))
        xs_div_exp_term_2 = np.divide(xs, 1 + np.exp(((2 * m2) / s2) * xs))

        xs_div_cosh_term_1 = np.divide(xs, 1 + np.cosh(((2 * m1) / s1) * xs)) 
        xs_div_cosh_term_2 = np.divide(xs, 1 + np.cosh(((2 * m2) / s2) * xs))

        xs2_div_cosh_term_1 = xs * xs_div_cosh_term_1
        xs2_div_cosh_term_2 = xs * xs_div_cosh_term_2 

        h11 = -np.sum(gs * ( (-1 / s1) + (2 / (s1 ** 2)) * xs2_div_cosh_term_1))

        h12 = -np.sum(gs * ( (xs - m1) / (-s1 ** 2) + (2 / (s1 ** 2)) * xs_div_exp_term_1 - ((2 * m1) / (s1 ** 3)) * xs_div_cosh_term_1))

        h22 = -np.sum(gs * ((1/(2 * s1 ** 2)) - ((xs - m1) ** 2) / (s1 ** 3) - (4 * m1 / (s1 ** 3)) * xs_div_exp_term_1 - (4 * (m1 ** 2) / (s1 ** 3) ) * xs2_div_cosh_term_1))

        h33 = -np.sum((1 - gs) * ( (-1 / s2) + (2 / (s2 ** 2)) * xs2_div_cosh_term_2))

        h34 = -np.sum((1 - gs) * ( (xs - m2) / (-s2 ** 2) + (2 / (s2 ** 2)) * xs_div_exp_term_2 - ((2 * m2) / (s2 ** 3)) * xs_div_cosh_term_2))

        h44 = -np.sum((1 - gs) * ((1/(2 * s2 ** 2)) - ((xs - m2) ** 2) / (s2 ** 3) - (4 * m2 / (s2 ** 3)) * xs_div_exp_term_2 - (4 * (m2 ** 2) / (s2 ** 3) ) * xs2_div_cosh_term_2))


        return np.array([
            [h11, h12, 0, 0], 
            [h12, h22, 0, 0],
            [0, 0, h33, h34],
            [0, 0, h34, h44]
        ]) 

    iters = 0

    old_gs = None
    while True:
    
        # expectation step 

        #compute probability each element of x belongs to distribution 1
        a = mix_p * fn_pdf(xs, mu_1, sig2_1)
        gs = a / (a + (1 - mix_p) * fn_pdf(xs, mu_2, sig2_2))


        #check if we have converged within the threshold, if so exit
        if old_gs is not None and np.allclose(gs, old_gs, atol=conv_tol):
            break
        iters += 1
        
        # maximisation step

        result = optimize.minimize(F, [mu_1, sig2_1, mu_2, sig2_2], args=(xs, gs), jac=True, hess=hess, method=method, options={"maxiter": 10000})

        mix_p = np.mean(gs)
        
        if not result.success:
            print(f"From optimizer: {result.message} (status: {result.status})")

        mu_1, sig2_1, mu_2, sig2_2 = result.x
        
        old_gs = gs


    print(iters)
    return mix_p, mu_1, mu_2, sig2_1, sig2_2


if __name__ == "__main__":
    
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
    print(f"params 1: (mu, sig) = ({true_m1}, {true_s1})")
    print(f"params 2: (mu, sig) = ({true_m2}, {true_s2})")
    print(f"mixing: {len(data_1) / data_size}")
    print()


    guess_m1 = abs(true_m1 + np.random.uniform(-0.1, 0.1))
    guess_m2 = abs(true_m2 + np.random.uniform(-0.1, 0.1))

    guess_s1 = abs(true_s1 + np.random.uniform(-0.1, 0.1))
    guess_s2 = abs(true_s2 + np.random.uniform(-0.1, 0.1))
    
    guess_mix_p = true_mix_p + np.random.uniform(-0.1, 0.1)

    print("Inital guesses:")
    print(f"params 1: (mu, sig) = ({guess_m1}, {guess_s1})")
    print(f"params 2: (mu, sig) = ({guess_m2}, {guess_s2})")
    print(f"mixing: {guess_mix_p}")
    print()


    mix_p, m1, m2, s1, s2 = em_2_folded_normals(data, guess_mix_p, guess_m1, guess_m2, guess_s1, guess_s2, conv_tol=1e-12, method="Nelder-Mead")

    print("Fitted values:")
    print(f"params 1: (mu, sig) = ({m1}, {s1})")
    print(f"params 2: (mu, sig) = ({m2}, {s2})")
    print(f"mixing: {mix_p}")

    line = np.linspace(min(data), max(data), num=1000)
    pdf_pts = mix_p * fn_pdf(line, m1, s1) + (1 - mix_p) * fn_pdf(line, m2, s2)

    guess_pts = guess_mix_p * fn_pdf(line, guess_m1, guess_s1) + (1 - mix_p) * fn_pdf(line, guess_m2, guess_s2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(line, pdf_pts, c='r')
    ax.plot(line, guess_pts, c='k', alpha=0.5)

    ax.hist(data, density=True, histtype='stepfilled', alpha=0.2)

    plt.show()
