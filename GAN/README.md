[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zWIWroxO)
# A Generative Model - Generative Adversarial Networks

## 1. Objective
- Generative model
- Adversarial learning

## 2. Baseline notebook code
- [assignment_08.ipynb](assignment_08.ipynb)

## 3. Data
- 100 images of size $32 \times 32$ 

## 4. Neural Network Architecture
- define a neural network architecture for a discriminator 
- define a neural network architecture for a generator 

## 5. Objective function
- Let $\theta$ be a set of model parameters for generator $G_\theta$
- Let $\phi$ be a set of model parameters for discriminator $D_\phi$
- min-max optimization
- adversarial optimization 
$$
\begin{align}
\mathcal{L}(\theta, \phi) &= \mathbb{E}_{x \sim p_{\rm data}} \log(D_\phi(x)) + \mathbb{E}_{z \sim \mathcal{N}(0, I)} \log(1 - D_\phi(G_\theta(z)))\\
\theta^* &= \arg \min_\theta \mathcal{L}(\theta, \phi)\\
\phi^* &= \arg \max_\phi \mathcal{L}(\theta, \phi)
\end{align}
$$

## 6. Submission
- you have to include the followings at your submission:
    - generator model: `generator.pth`
    - discriminator model: `discriminator.pth`
    - result file: `result.csv`
        - FID
        - generator loss
        - discriminator loss

## 7. Codes for testing
- `tests/prepare_test.sh`
- `tests/run_test.py`
- `tests/eval_test.py`

## 8. Evaluation
- the accuracy is computed in terms of FID 
- the score will be given by 100 - FID

## 9. Protected folders
- You should not modify anything under the following protected folders:
- Modification will be indicated at GitHub Classroom, which will lead to Fail
```console
.github/workflows
tests
```
