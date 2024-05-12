# Speeding up Monte Carlo Integration: Control Neighbors for Optimal Convergence

This is the code for the article "Speeding up Monte Carlo Integration: Control Neighbors for Optimal Convergence", by Rémi LELUC, François PORTIER, Johan SEGERS and Aigerim ZHUMAN. [arXiv](https://arxiv.org/pdf/2305.06151.pdf). To appear in [Bernoulli (2024)](https://www.bernoullisociety.org/publications/bernoulli-journal/bernoulli-journal-papers)

This implementation is made by [Rémi Leluc](https://remileluc.github.io/)

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

## Citation

> @article{leluc2023speeding,
  title={Speeding up Monte Carlo Integration: Control Neighbors for Optimal Convergence},
  author={Leluc, R{\'e}mi and Portier, Fran{\c{c}}ois and Segers, Johan and Zhuman, Aigerim},
  journal={arXiv preprint arXiv:2305.06151},
  year={2023}
}
>
