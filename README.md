# FEEMS setup

**F**ast **E**stimation of **E**ffective **M**igration **S**urfaces (`feems`) is a python package 
implementing a statistical method for inferring and visualizing gene-flow in 
spatial population genetic data.

The `feems` method and software was developed by Joe Marcus and Wooseok Ha and 
advised by Rina Foygel Barber and John Novembre. We also used code from Benjamin M. Peter 
to help construct the spatial graphs. 

For details on the method see our [pre-print](https://www.biorxiv.org/content/10.1101/2020.08.07.242214v1). Note that `feems` is in review so the method could be subject to change.  

# Setup

We've found that the easiest way to get started is to setup a `conda` 
environment:

```
conda create -n=feems_e python=3.8.3 
conda activate feems_e
```

Some of the plotting utilities in the `feems` package require `geos` as a 
dependency which can be installed on mac with brew as follows:

```
brew install geos
```

Unfortunately some of the other dependencies for `feems` are not easily 
installed by pip so we recommend getting started using `conda`:

```
conda install -c conda-forge suitesparse=5.7.2 scikit-sparse=0.4.4 cartopy=0.18.0 jupyter=1.0.0 jupyterlab=2.1.5 sphinx=3.1.2 sphinx_rtd_theme=0.5.0 nbsphinx=0.7.1 sphinx-autodoc-typehints
```

We added jupyter and jupyterlab to explore some example notebooks but these 
are not necessary for the `feems` package. Once the `conda` environment has 
been setup with these tricky dependencies we can install `feems`:

```
pip install git+https://github.com/jhmarcus/feems
```

You can also install `feems` locally by:

```
git clone https://github.com/jhmarcus/feems
cd feems/
pip install .
```

# Spatial Prediction with FEEMS

This document is intended to detail the rotation project I worked on with John Novembre in Summer 2021.  If you're interested in spatial prediction you might also want to check out: https://github.com/karltayeb/spatial_prediction_workflow

## Methods
### Brief model review
Let ${\bf f}_j$ be a vector denoting the true allele frequency of SNP $j$ across $d$ demes.
Let $\hat{\bf f}_j$ denote the observed allele frequency at a subset of $o$ demes.
Let $A \in \{0, 1\}^{o \times d}$ be a matrix such that each row selects one of the observed demes.
$$
\begin{align}
\hat{\bf f}_j | {\bf f}_j \sim \mathcal N_o \left(A{\bf f}_j, \mu_j(1-\mu_j){\bf d}\right)\\
{\bf f}_j \sim \mathcal N_d(\mu_j {\bf 1}, \mu_j(1-\mu_j)L^{\dagger}_{dd})
\end{align}
$$

True allele frequencies are modeled as multivariate normal, with a covariance matrix given by $\mu_j(1-\mu_j) L_{dd}^{\dagger}$ .

Observed allele frequencies, conditioned on true allele frequency, are observed with independent sampling error. Assuming Hardy Weinberg, the alternate allele count at a deme would be binomially distributed, which is well approximated by a normal when sample size is large and minor allele frequency is not too close 0. Note that this would suggest $\hat{\bf f}_j | {\bf f}_j \sim \mathcal N_o \left(A{\bf f}_j, diag({\bf f}_j(1-{\bf f}_j)){\bf d}\right)$ but for computational reasons $f_j(1-f_j)$ is replaced with the global mean allele frequency.

The psuedoinverse of the graph Laplacian $L_{dd}^{\dagger}$ gives the expected commute times between demes $i$ and $j$ on the graph by $L_{ii}^{\dagger} + L_{jj}^{\dagger}- 2L_{ij}^{\dagger}$, which approximates coalescence times in the stepping model (and by extention, the covariance between allele frequencies at these two demes). Intuitively if two demes are far apart (long commute time) there is not much gene flow between them, covariance is low, and the allele frequencies will evlolve nearly independently. If demes are close together (short commute times on the graph) there is gene flow, covariance is high.

### Posterior allele frequency derivation

The observed and latent allele frequencies are jointly gaussian, we can use this fact to arrive at the posterior allele frequncy at all demes, given the allele frequency at observed demes

$$
\begin{align}
{\bf f}_j | \hat{\bf f}_j \sim \mathcal N_d(\bar \mu^{(j)}, \bar \Sigma^{(j)}) \\
\bar \mu^{(j)} = \mu_j {\bf 1} + L_{do}^{\dagger}(L_{oo}^{\dagger} + \sigma^2 {\bf d})^{-1}(\hat {\bf f}_j - \mu_j{\bf 1}_o)\\
\bar \Sigma^{(j)} = \mu_j(1-\mu_j)\left[L_{dd}^{\dagger}(L_{oo}^{\dagger} + \sigma^2 {\bf d})^{-1}L_{od}^{\dagger}\right]
\end{align}
$$

### Extending FEEMS to spatial prediction

Armed with a fit FEEMS model and a new sample of unkown origin with genotype ${\bf g}$, our problem is to determine (ideally with well calibrated uncertainty) the origin of the sample. Let the (unobserved) deme of origin for the sample be $z$

For each SNP $i$ we can evaluate the likelihood of observing an individual with $g_i$ copies of the alternate allele in deme $k$ as

$$
\begin{align}
g_i | f_{ik} \sim \text{Binomial}(g_i | 2, f_{ik}))\\
f_{ik} \sim \mathcal{N}(\bar\mu_{k}^{(i)}, \bar \Sigma_{kk}^{(i)})
\end{align}
$$

$f_{ik}$ the allele frequency for SNP $i$ in deme $k$ is given by the marginal posterior distribution from FEEMS. We need to compute/approximate the probability that genotype $g_i$ is observed in deme $k$, marginalizing out $f_{ik}$. There is a small wrinkle here-- the normal distribution is not constrained to the interval $(0, 1)$  so it won't always give a valid allele frequency. We have a few options

1. Truncate the normal distribution to the unit interval (what we currently do)
2.  Provide some other transformation of the normal to the unit interval, preferably one that does not distort valid values.
3. Match the moments of the normal to a distribution with unit support (e.g beta, so that you get a beta-binomial). This idea seems elegant, but in the case of the beta distribution does not work well when the moments of the normal imply a bimodal (U-shaped) beta.


However you do it we'll assume from here you can compute $p(g_i | z = k)$. Assuming all the observed alleles in ${\bf g}$ are independent we have 
$$p_k = p({\bf g} | z=k) = \prod_i p(g_i | z=k)$$

To perform assignment we can specify a prior over demes $\bf \pi$ giving the prior probability of sampling an individual from each deme. The posterior assignments are given by
$$ p(z=k | {\bf g}) = \frac{p_k \pi_k}{\sum_j p_j \pi_j}$$
## Implimentation notes

**You only need the diagonal of the posterior variance** For spatial prediction you only need the marginal posteriors. This means that if you can find a way to only compute the diagonal of the posterior variance you may improve the memory/time efficiency of the problem. In the experiments we looked at we used the posterior mean allele frequencies as point estimates rather than integrating over the posterior.

All code specific to spatial prediction is found in `feems/spatial_prediction.py`. Comments and docstrings should be helpful in understanding how it works, but also all the main functionality is summarized here.

Breifly :
- `_compute_frequency_posterior` computes the FEEMS posterior mean, and optionally the posterior variance, of allele frequencies across demes/SNPS
- `_compute_assignment_probabilities_point_mu` and `_compute_assignment_probabilities_trunc_normal`  compute the posterior probability of assignment, using either point estimates of allele frequencies or integrating over truncated posterior respectively. If you do not provide a prior over demes we assume individuals are sampled with equal probability from all demes.
- `predict_deme_point_mu` and `predict_deme_trunc_normal_mu` wrap the `_compute_assignment_{method}` functions, return the posterior assignments AND posterior mean of allele frequencies. If you do not provide a prior over demes we assume individuals are sampled with equal probability from all demes.
- `leave_node_out_spatial_prediction` and `predict_held_out_samples` are high level functions for testing the spatial prediction using samples with known sampling lcoation.

