from .objective import Objective, comp_mats
from .spatial_graph import query_node_attributes
from .cross_validation import train_test_split

import numpy as np
from scipy.stats import norm


def predict_held_out_nodes(sp_graph, coord, predict_type='point_mu', fit_feems=True, fit_kwargs={}):
    sample_idx = query_node_attributes(sp_graph, 'sample_idx')
    permuted_idx = query_node_attributes(sp_graph, "permuted_idx")


    sp_graph.fit_null_model()
    
    # deepcopy doesn't like sp_graph.factor...
    sp_graph.factor = None
    
    # remove test demes from training
    n = sp_graph.sample_pos.shape[0]
    split = ~np.isnan(coord.iloc[:, 0])
    sp_graph_train, sp_graph_test = train_test_split(sp_graph, split)
    

    test_sample_idx = query_node_attributes(sp_graph_test, 'sample_idx')
    test_node2sample = {i: test_sample_idx[i]
        for i in range(len(test_sample_idx))
        if len(test_sample_idx[i]) > 0}
    test_nodes = list(test_node2sample.keys())
    print('fit feems w/o observations @ node: {}'.format(test_nodes))
    
    if fit_feems:
        # TODO use fit_kwargs
        # sp_graph_train.fit(**fit_kwargs)
        sp_graph_train.fit(lamb=2., verbose=True)


    # get genotypes of test deme
    g = sp_graph.genotypes
    g[~np.isclose(g, g.astype(int))] = np.nan
    
    # predict
    if predict_type == 'point':
        z, post_mean = predict_deme_point_mu(g, sp_graph_train)

    # predict
    if predict_type == 'trunc':
        z, post_mean = predict_deme_trunc_normal_mu(g, sp_graph_train)

    results = {
        'post_assignment': z,
        'w': sp_graph_train.w,
        'w0': sp_graph_train.w0,
        's2': sp_graph_train.s2,
        'post_mean': post_mean, # compute posterior mean
        'map_coord': sp_graph.node_pos[permuted_idx][z.argmax(1)],
        'pred_idx': np.where(~split)[0]
    }
    return results


def leave_node_out_spatial_prediction(sp_graph, predict_type='point_mu', fit_feems=True, fit_kwargs={}, max_nodes=500):
    sample_idx = query_node_attributes(sp_graph, 'sample_idx')
    permuted_idx = query_node_attributes(sp_graph, "permuted_idx")

    node2sample = {i: sample_idx[i] for i in range(len(sample_idx))}
    obsnode2sample = {
        k: v for k, v in node2sample.items() if len(v) > 0
    }
    sp_graph.fit_null_model()
    results = {}

    for node, samples in list(obsnode2sample.items())[:max_nodes]:
        print('fit feems w/o observations @ node: {}'.format(node))
        # deepcopy doesn't like sp_graph.factor...
        sp_graph.factor = None
        
        # remove deme from training
        n = sp_graph.sample_pos.shape[0]
        split = ~np.isin(np.arange(n), samples)
        sp_graph_train, sp_graph_test = train_test_split(sp_graph, split)
        
        
        if fit_feems:
            # TODO use fit_kwargs
            # sp_graph_train.fit(**fit_kwargs)
            sp_graph_train.fit(lamb=20., verbose=False)


        # get genotypes of test deme
        g = sp_graph_test.genotypes
        g[~np.isclose(sp_graph_test.genotypes, sp_graph_test.genotypes.astype(int))] = np.nan
        
        # predict
        if predict_type == 'point':
            z, post_mean = predict_deme_point_mu(g, sp_graph_train)

        # predict
        if predict_type == 'trunc':
            z, post_mean = predict_deme_trunc_normal_mu(g, sp_graph_train)

        sp_graph_train.factor = None


        results[node] = {
            'post_assignment': z,
            'w': sp_graph_train.w,
            'w0': sp_graph_train.w0,
            's2': sp_graph_train.s2,
            #'post_mean': post_mean, # compute posterior mean
            'true_coord': sp_graph.sample_pos[sample_idx[node]],
            'map_coord': sp_graph.node_pos[permuted_idx][z.argmax(1)]
        }
    return results

def logsumexp(x):
    x = np.atleast_2d(x)
    c = x.max(1)
    return c + np.log(np.sum(np.exp(x - c[:, None]), 1))

def _compute_assignment_probabilities_point_mu(g, f, eps=1e-5):
    f_clip = np.clip(f, eps, 1-eps)
    lp = np.log(f_clip)
    lq = np.log(1 - f_clip)
    g = np.atleast_2d(g)
    c = np.log(np.isclose(g, 1) + 1)  # (2 choose g)

    z = (lp @ np.nan_to_num(g.T) + lq @ np.nan_to_num(2 - g.T)).T
    z = z - logsumexp(z)[:, None]
    return z

def _truncated_moments(mu, scale, a=0, b=1):
    a = - mu / scale
    b = (1 - mu) / scale
    Z = norm.cdf(b) - norm.cdf(a)
    
    phi_a, phi_b = norm.pdf(a), norm.pdf(b)
    
    m1 = mu + (phi_a - phi_b) / Z * scale

    var = scale**2 * (1 + (a * phi_a - b * phi_b) / Z) \
        - scale * (m1 - mu)**2
    return m1, var

def _compute_assignment_probabilities_trunc_normal(g, mu, var, eps=1e-5):
    f, v = _truncated_moments(mu, np.sqrt(var))
    f2 = v + f ** 2
    z = (np.log(f2) @ (g.T == 2) +
        np.log(2 * (f - f2)) @ (g.T == 1) +
        np.log(1 - 2*f + f2) @ (g.T == 0)).T
    z = z - logsumexp(z)[:, None]
    return z

def _compute_frequency_posterior(g, sp_graph_train, compute_var=False):
    """
    compute posterior distribution of latent allele frequences
    """

    d = len(sp_graph_train)
    o = sp_graph_train.n_observed_nodes

    obj = Objective(sp_graph_train)
    obj.sp_graph.comp_graph_laplacian(sp_graph_train.w)

    # make sure we need all of theses calls to run comp_mats(obj)
    obj._solve_lap_sys()
    obj._comp_mat_block_inv()
    obj._comp_inv_cov()
    obj._comp_inv_lap()
    fit_cov, inv_cov, _ = comp_mats(obj)


    frequencies_ns = sp_graph_train.frequencies * np.sqrt(sp_graph_train.mu * (1 - sp_graph_train.mu))
    mu0 = frequencies_ns.mean(axis=0) / 2
    mu_f = np.sqrt(sp_graph_train.mu * (1 - sp_graph_train.mu))

    frequencies = sp_graph_train.frequencies
    scale = mu_f / 2
    mu_frequencies = mu0 / scale

    Linv = obj.Linv - 1 / d
    post_mean = mu_frequencies + Linv @ inv_cov @ (frequencies - mu_frequencies)
    post_mean = post_mean * scale

    if compute_var:
        # TODO: impliment fast approximation to compute L_pinv_diag
        L_dense = sp_graph_train.L.todense()
        L_pinv_diag = np.diag(np.linalg.pinv(L_dense))
        post_var = L_pinv_diag - np.einsum('ij,ji->i', Linv, inv_cov @ Linv.T)
        post_var = post_var[:, None] * (scale[None] ** 2)
    else:
        post_var = None
    return post_mean, post_var

def predict_deme_point_mu(g, sp_graph_train):
    # get genotype from sp_graph_test
    post_mean, _ = _compute_frequency_posterior(g, sp_graph_train)
    z = _compute_assignment_probabilities_point_mu(g, post_mean)
    return z, post_mean

def predict_deme_trunc_normal_mu(g, sp_graph_train):
    # get genotype from sp_graph_test
    post_mean, post_var = _compute_frequency_posterior(g, sp_graph_train, compute_var=True)
    z = _compute_assignment_probabilities_trunc_normal(g, post_mean, post_var)
    return z, post_mean

def predict_deme_beta_mu(sp_graph_train, sp_graph_test):
    pass