[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ZzINU6yu)
# A multi-label classification based on Pytorch

## 1. Objective

- Dataset
- Neural Network
- Objective Function
- Optimization

## 2. Baseline notebook code

- [assignment_03.ipynb](assignment_03.ipynb)

## 3. Utility codes

#### You have to complete the following codes:

- [MyModel.py](MyModel.py)
- [MyTrain.py](MyTrain.py) : you can create a class for training your model

#### You do not have to modify the following codes:

- [MyDataset.py](MyDataset.py)
- [MyEval.py](MyEval.py)
- [MyResult.py](MyResult.py)

## 4. Data

- mnist images of size $16 \times 16$ for digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
  - train ($100 \times 10$ classes)
  - test ($850 \times 10$ classes)

## 5. Neural Network Architecture (`MyModel.py`)

- you can use pre-defined layer functions and activation functions
- you can use package `import torch.nn as nn`
- you can use convolution layters `nn.Conv2d`
- you can use multiple layers for the feature characterization
- you can use multiple layers for the classification
- you can use any initialization method for the model

## 6. Objective function

- you can use pre-defined objective function such as `nn.BCEWithLogitsLoss`, `nn.CrossEntropyLoss`, `nn.Softmax`
- you can use back-propagation function `backward`

## 7. Optimization function

- you can use ppackage `import torch.optim as optim`
- you can use stochastic gradient descent function `optim.SGD`
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

- the score will be given by the average of the training accuracy and the testing accuracy
- the testing accuracy will be computed at the Autograding plateform of GitHub Classroom
- the mock test will be performed in the evening of the day before the next class date
- questions related to the assignment are recommended to be asked during the following class

## Protected folders
- You should not modify anything under the following protected folders:
- Modification will be indicated at GitHub Classroom, which will lead to Fail
```console
.github/workflows
tests
data
```