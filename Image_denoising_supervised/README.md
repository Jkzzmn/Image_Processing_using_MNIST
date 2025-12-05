[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/x7a6ye_P)
# Image denoising by an unsupervised learning

## 1. Objective
- Dataset Manipulation
- Neural Network Architecture
- Objective Function
- Optimization
- Learning-rate Scheduling

## 2. Baseline notebook code
- [assignment_05.ipynb](assignment_05.ipynb)

## 3. Utility codes
#### You have to complete the following codes:
- [MyDataset.py](MyDataset.py) : you need to define your dataset for training
- [MyModel.py](MyModel.py)
- [MyTrain.py](MyTrain.py) : you can create a class for training your model

#### You do not have to modify the following codes:
- [MyEval.py](MyEval.py)
- [MyResult.py](MyResult.py)

## 4. Data
- dataset consists of `train`, `val`, and `test`
- `train` and `val` are provided at training, but `test` is only available at evaluation
- `val` and `test` follow the same data distribution
- `train` does not follow the same data distribution as `val` or `test` 
- you should use ONLY the training dataset for training your model
- you should NOT use any other data including `val` dataset other than the provided `train` dataset when training your model
- `val` dataset is only for the validation of your model and the choice of hyper-parameters
- `data` is a protected folder and you should not modify any contents under `data` folder

#### Training dataset
- mnist images of size $16 \times 16$ for the class of $0, 1, 2, 3$
- $100 \times 4$ classes

#### Validation dataset
- mnist images of size $16 \times 16$ for the class of $0, 1, \cdots, 9$
- $10 \times 10$ classes

#### Testing dataset
- mnist images of size $16 \times 16$ for the class of $0, 1, \cdots, 9$
- $1500 \times 10$ classes

#### Data augmentation (`MyDataset.py`)
- you can apply any data augmentation at training in order to improve the generalization of your model

## 5. Neural Network Architecture (`MyModel.py`)
- you should construct your model manually
- you should not load a pre-defined model architecture from the model zoo
- you can use package `import torch.nn as nn`
- you can use pre-defined layer functions and activation functions
- the neural network architecture is constructed to be an auto-encoder
- the dimensions of input and output of model should be the same
- an auto-encoder generally consists of an encoder and a decoder through a bottleneck 
- you can use multiple layers for the encoder 
- you can use multiple layers for the decoder
- you can use any initialization method for the model

## 6. Objective function (`MyTrain`)
- you can use pre-defined objective function such as `torch.nn.MSELoss`, `torch.nn.L1Loss`, `torch.nn.HuberLoss` or any other loss including percepture losses
- you can use back-propagation function `backward` of the defined loss
- you can consider an objective function that consists of data fidelity and regularization

#### image model
>$$
\begin{aligned}
f(x) = u(x) + \eta(x), \quad \eta(x) \sim \mathcal{N}(0, \sigma^2)
\end{aligned}
>$$

#### prediction
>$$
h = \phi_\theta(f)
>$$
>where $\phi_\theta$ is a model associated with a set of model parameters $\theta$

#### data fidelity
>$$
\begin{aligned}
\int_\Omega \| h(x) - f(x) \|_2^2 \, dx,
\end{aligned}
>$$
>where $\Omega$ denotes the spatial domain of data

#### regularization
>$$
\begin{aligned}
\int_\Omega \| \nabla h(x) \|_2^2 \, dx
\end{aligned}
>$$
>where $\nabla$ denotes the spatial gradient operator

You can also consider the regularization of the model parameters as given by:
$$
\| \theta \|_2^2
$$

## 7. Optimization function (`MyTrain`)
- you can use ppackage `import torch.optim as optim`
- you can use any optimizer such as `optim.SGD`, `optim.Adam`, `optim.AdamW` or any other optimizer
- you can use a scheduling method for the learning rate
- you can use your own hypter-parameters for the optimization such as `batch size`, `learning rate`, `number of epochs`

## Procedure
### (1) Clone a repository for the assignment from GitHub to local machine
```console
$ git clone path_to_repository
```

### (2) Set up a Python Virtual Environment 
#### Create a virtual environment
```console
$ python -m venv env_OO
```

#### Activate a virtual environment
```console
$ source env_OO/bin/activate
```

#### Update `pip`
```console
$ pip install --upgrade pip
```

#### Install required packages defined in `requirements.txt`
```console
$ pip install -r requirements.txt
```

### (3) Submission
- `commit` should be made at least 30 times
- the message of `commit` should be informative on the working progress
- the history of `commit` should effectively indicate the coding progress
- deadline is 23:30 on the first Sunday after the following class date
- you do not have to submit anything to Google Classroom
  
#### Add all the files in the assignment repository (e.g. `model.pth`, `result.csv`)
```console
$ git add .
```

#### Commit the codes
```console
$ git commit -am "commit message"
```

### Push all the codes in the repository of local machine to GitHub
```console
$ git push origin main
```

### Deactivate the virtual environment
```console
$ deactivate
```

### (4). Evaluation
- the accuracy is computed in terms of PSNR as defined by:
>$$
\begin{align}
PSNR(I, J) &= 10 \cdot \log_{10} \frac{\max{(I)}^2}{MSE(I, J)}\\
MSE(I, J) &= \frac{1}{n} \sum_{i=1}^n (I_i - J_j)^2
\end{align}
>$$
>where $n$ denotes the number of elements in $I$ and $J$

- higher PSNR values generally indicating better reconstruction quality
- training accuracy is given by the training PSNR + 50.0
- testing accuracy is given by the testing PSNR + 50.0
- the score will be given by the weighted average of the training accuracy (0.3) and the testing accuracy (0.7)
- the testing accuracy will be computed at the Autograding plateform of GitHub Classroom using the test dataset that is unavailable at training
- the mock test based on both training and testing datasets will be performed in the evening of the day before the next class date
- questions related to the assignment are recommended to be asked during the following class

## Protected folders
- You should not modify anything under the following protected folders:
- Modification will be indicated at GitHub Classroom, which will lead to Fail
```console
.github/workflows
tests
data
```