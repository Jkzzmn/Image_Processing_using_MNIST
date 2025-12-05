[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ECLUHhQC)
# A Generative Model - Variational Autoencoder

## 1. Objective
- Generative model
- Variational Approach

## 2. Baseline notebook code
- [assignment_09.ipynb](assignment_09.ipynb)

## 3. Data
- 100 images of size $1 \times 32 \times 32$ 

## 4. Neural Network Architecture
- define a neural network architecture for an encoder
- define a latent class for the reparameterization
- define a neural network architecture for a decoder 

## 5. Generative Model

- observed data: $x \in X$
- unobserved data: $z \in Z$
- prior probability: $p(x)$
- joint probability: $p(x, z)$
- likelihood probability: $p(x \vert z)$
- posterior probability: $p(z \vert x)$
- objective function is designed to approximate the posterior distributio $p$ with a parameterized approximate distribution $q$ as follows:
$$
\begin{align}
q(z \vert x) \approx p(z \vert x)
\end{align}
$$

## 6. Objective function
- [tutorial](https://arxiv.org/pdf/1907.08956)
- Let $\theta$ be a set of model parameters for encoder $E_\theta$
- Let $\phi$ be a set of model parameters for decoder $D_\phi$

### Kullback-Leibler divergence (KL divergence)
$$
KL(P \| Q) = \sum_{x \in X} P(x) \log\left( \frac{P(x)}{Q(x)} \right)
$$

### evidence lower bound (ELBO)
$$
\begin{align}
KL(q(z \vert x) \| p(z \vert x)) &= \log p(x) + \mathbb{E}_{z \sim q(\cdot \vert x)} \left[ \log \left( \frac{q(z \vert x)}{p(x, z)} \right) \right] \ge 0\\
\log p(x) &\ge -\mathbb{E}_{z \sim q(\cdot \vert x)} \left[ \log \left( \frac{q(z \vert x)}{p(x, z)} \right) \right] 
\end{align}
$$

### maximizing ELBO

- maximizing the evidenice is equivalent to minimizing $\mathbb{E}_{z \sim q(\cdot \vert x)} \left[ \log \left( \frac{q(z \vert x)}{p(x, z)} \right) \right]$:

$$
\begin{align}
\mathbb{E}_{z \sim q(\cdot \vert x)} \left[ \log \left( \frac{q(z \vert x)}{p(x, z)} \right) \right] 
&= - \mathbb{E}_{z \sim q(\cdot \vert x)} \left[ \log \left( p(x \vert z) \right) \right] + \mathbb{E}_{z \sim q(\cdot \vert x)} \left[ \log \left( \frac{q(z \vert x)}{p(z)} \right) \right]  

\end{align}
$$


### Data Fidelity term (reconstruction loss)
$$
- \mathbb{E}_{z \sim q(\cdot \vert x)} \left[ \log p(x \vert z) \right] = \frac{1}{2} \mathbb{E}_{z \sim q(\cdot \vert x)} \left[ \| x - D_\phi(z) \|_2^2 \right] 
$$

### Regularization term (Kullback-Leibler divergence) ($\sigma_p = 1, \mu_p = 0$)

$$
\begin{align}
\mathbb{E}_{z \sim q(\cdot \vert x)} \left[ \log \left( \frac{q(z \vert x)}{p(z)} \right) \right] 
&= - \frac{1}{2} \left( 1 + \log \left( \sigma_q^2 \right) - \sigma_q^2 - \mu_q^2 \right)
\end{align}
$$
where 
$$
E_\theta(x) = 
    \left[
        \begin{array}{c}
            \mu \\
            \log(\sigma^2)
        \end{array}
    \right]
$$
- note: it is numerically stable for Encoder to predict $(\log (\sigma^2), \mu)$ instead of $(\sigma, \mu)$

## 7. Submission
- you have to include the followings at your submission:
    - encoder model: `encoder.pth`
    - latent model: `latent.pth`
    - decoder model: `decoder.pth`
    - result file: `result.csv`
      - `FID`: FID computed by the evaluation code
      - `loss`: loss per epoch (average over batches within each epoch or at the last batch)
    - notebook: 
      - `sample`: generated samples 

## 8. Codes for testing
- codes used for testing in `tests` folder
```
tests/prepare_test.sh
tests/run_test.py
tests/eval_test.py
```
- preparation code before testing:
```
rm -f result_autograding.csv
rm -rf data/sample
mkdir -p data/sample
``` 
- command for self-testing: 
```
python -m unittest tests/run_test.py
```

## 9. Evaluation
- the accuracy is computed in terms of FID 
- the score will be given by 100 - FID

## 10. Protected folders
- You should not modify anything under the following protected folders:
- Modification will be indicated at GitHub Classroom, which will lead to Fail
```console
.github/workflows
tests
```