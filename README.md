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

A novel linear integration rule called *control neighbors* is proposed in which nearest neighbor estimates act as control variates to speed up the convergence rate of the Monte Carlo procedure on metric spaces. The main result is the $\mathcal{O}(n^{-1/2} n^{-s/d})$ convergence rate -- where $n$ stands for the number of evaluations of the integrand and $d$ for the dimension of the domain -- of this estimate for Hölder functions with regularity $s \in (0,1]$, a rate which, in some sense, is optimal. Several numerical experiments validate the complexity bound and highlight the good performance of the proposed estimator.

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


