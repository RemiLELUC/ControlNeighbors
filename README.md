# Speeding up Monte Carlo Integration: Control Neighbors for Optimal Convergence

This is the code for the article "Speeding up Monte Carlo Integration: Control Neighbors for Optimal Convergence", by Rémi LELUC, François PORTIER, Johan SEGERS and Aigerim ZHUMAN. ([arXiv](https://arxiv.org/pdf/2305.06151.pdf),[download paper](https://www.e-publications.org/ims/submission/BEJ/user/submissionFile/62730?confirm=5788f3a9)). To appear in [Bernoulli (2024)](https://www.bernoullisociety.org/publications/bernoulli-journal/bernoulli-journal-papers).

This implementation is made by [Rémi Leluc](https://remileluc.github.io/).

## Citation

> @article{leluc2023speeding,
  title={Speeding up Monte Carlo Integration: Control Neighbors for Optimal Convergence},
  author={Leluc, R{\'e}mi and Portier, Fran{\c{c}}ois and Segers, Johan and Zhuman, Aigerim},
  journal={arXiv preprint arXiv:2305.06151},
  year={2023}
}
>

## Abstract

A novel linear integration rule called **control neighbors** is proposed in which nearest neighbor estimates act as control variates to speed up the convergence rate of the Monte Carlo procedure on metric spaces. The main result is the $\mathcal{O}(n^{-1/2} n^{-s/d})$ convergence rate -- where $n$ stands for the number of evaluations of the integrand and $d$ for the dimension of the domain -- of this estimate for Hölder functions with regularity $s \in (0,1]$, a rate which, in some sense, is optimal. Several numerical experiments validate the complexity bound and highlight the good performance of the proposed estimator.

## Control Neighbors estimate

Consider the classical numerical integration problem of approximating the value of an integral 
$$\mu(\varphi) = \int \varphi \mathrm{d} \mu$$
where $\mu$ is a probability measure on a metric space $(M, \rho)$ and the integrand $\varphi$ is a real-valued function on the support of $\mu$. Suppose that random draws from the measure $\mu$ are available and calls to the function $\varphi$ are possible. The standard Monte Carlo estimate consists in averaging $\varphi(X_i)$ over $i=1,\ldots,n$, where the particles $X_i$ are drawn independently from $\mu$. 

For square-integrable integrands, the Monte Carlo estimate has convergence rate $\mathcal{O}(n^{-1/2})$ as $n \to \infty$, whatever the dimension of the domain.

A new Monte Carlo method called **control neighbors** is introduced. By using $1$-nearest neighbor estimates as control variates, it produces an estimate $\hat{\mu}_n(\varphi)$ of the integral $\mu(\varphi)$ for a probability measure $\mu$ on a metric space $(M, \rho)$ such that the measure of a ball of radius $r > 0$ is of the order $r^d$ as $r \to 0$, uniformly over the space. 

This novel estimate is shown to achieve the (faster) convergence rate $\mathcal{O}(n^{-1/2} n^{-s/d})$ for Hölder integrands of regularity $s \in (0, 1]$. It is given by

$$\mu_n^{\text{(NN)}}(\varphi) = \frac{1}{n} \sum_{i=1}^n \varphi(X_i) - \big( \hat \varphi\_n^{(i)}(X_i) -   \mu ( \hat{\varphi}\_n  ) \big)$$
where $\hat \varphi\_n^{(i)}$ is the $1$-NN leave-one-out estimate. It is piece-wise constant on Voronoi cells (see Figures below).

The method takes the form of a linear integration rule $\sum\limits_{i=1}^n w_{i,n} \varphi(X_i)$ with weights $w_{i,n}$ **not depending on the integrand** $\varphi$ but only on the particles $X_1, \ldots, X_n$. This property is computationally beneficial when several integrals are to be computed with the same measure $\mu$.

**Voronoi cells on unit square $[0,1]^2$**

<img src="https://github.com/RemiLELUC/ControlNeighbors/blob/master/graphs/voronoi_square.png"  width="60%" height="30%">

**Voronoi cells on unit sphere $\mathbb{S}^2$**

<img src="https://github.com/RemiLELUC/ControlNeighbors/blob/master/graphs/voronoi_sphere.png"  width="65%" height="35%">


## Description

Dependencies in Python 3
- requirements.txt : dependencies

Folders
- 1_Synthetic/:

contains the source code and results of the numerical experiments related to integration problems on the unit cube $[0,1]^d$, on $\mathbb{R}^d$, on the orthogonal group $O_m(\mathbb{R})$ and on the sphere $\mathbb{S}^{q-1}$.

- 2_OptionPricing/:

contains the source code and results of the numerical experiments related to option pricing with Black-Scholes model and Heston model.

- 3_OptimalTransport/:

contains the source code and results of the numerical experiments related to optimal transport with the computation of the Sliced-Wasserstein distance.

- graphs/:

contains all the figures.pdf related to the article.


