[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ZBD_MWde)
# Image segmentation by a supervised learning

## 1. Objective
- Dataset Manipulation
- Neural Network Architecture
- Objective Function
- Optimization
- Learning-rate Scheduling

## 2. Baseline notebook code
- [assignment_06.ipynb](assignment_06.ipynb)

## 3. Utility codes
#### You have to complete the following codes:
- [MyDataset.py](MyDataset.py) : you need to define your dataset for training
- [MyModel.py](MyModel.py) : you need to define your model architecture
- [MyTrain.py](MyTrain.py) : you can create a class for training your model

#### You do not have to modify the following codes:
- [MyEval.py](MyEval.py)
- [MyResult.py](MyResult.py)

## 4. Data
- dataset consists of `train`, `val`, and `test` datasets
- each dataset consists of a pair of (image, mask)
- `train` and `val` are provided at training, but `test` is only available at evaluation
- `val` and `test` follow the same data distribution
- `train` does not follow the same data distribution as `val` or `test` 
- you should use ONLY the training dataset for training your model
- you should NOT use any other data including `val` dataset other than the provided `train` dataset when training your model
- `val` dataset is only for the validation of your model and the choice of hyper-parameters
- `data` is a protected folder and you should not modify any contents under `data` folder

#### Training dataset
- mnist images with noises ($\sigma = 1.0$) and their segmentation masks of size $32 \times 32$ for the class of $0, 1, 2, 3$
- $100 \times 4$ classes

#### Validation dataset
- mnist images with noises ($\sigma = 1.0$) and their segmentation masks of size $32 \times 32$ for the class of $0, 1, \cdots, 9$
- $10 \times 10$ classes

#### Testing dataset
- mnist images with noises ($\sigma = 1.0$) and their segmentation masks of size $32 \times 32$ for the class of $0, 1, \cdots, 9$
- $1500 \times 10$ classes

#### Data augmentation (`MyDataset.py`)
- you can apply any data augmentation at training in order to improve the generalization of your model
- you may want to define your transformations for the image and the masks seperately

## 5. Neural Network Architecture (`MyModel.py`)
- you should construct your own model manually
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
\int_\Omega \phi(x) \, | f(x) - a |^2 + (1 - \phi(x)) \, | f(x) - b |^2 dx,
\end{aligned}
>$$
> where $\phi$ represents a segmenting function, $a$ indicates a representative constant for the inside of segmenting function $\phi$, and $b$ indiates a representative constant for the outside of segmenting function
> 
#### prediction
>$$
h = \phi_\theta(f)
>$$
>where $\phi_\theta$ is a model associated with a set of model parameters $\theta$, and $h$ is in the form of binary function

#### data fidelity
>$$
\begin{aligned}
\int_\Omega \| h(x) - m(x) \|_2^2 \, dx,
\end{aligned}
>$$
>where $\Omega$ denotes the spatial domain of data, and $m$ represents binary mask of $f$

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
- the accuracy is computed in terms of Intersection (IoU) over Union as defined by:
>$$
\begin{align}
IoU(I, J) &= \frac{|\, I \cap J \,|}{|\, I \cup J \,|}\\
\end{align}
>$$

- 1.0 indicates perfect matching between prediction and mask
- training accuracy is given by the training IoU * 100
- testing accuracy is given by the testing IoU * 100
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