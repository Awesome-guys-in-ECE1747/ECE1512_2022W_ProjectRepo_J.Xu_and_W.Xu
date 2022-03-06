<div align="center">

# ECE1512_2022W_ProjectRepo_J.Xu_and_W.Xu

</div>

<div align="center">

*The Project Repository of ECE1512 2020W*

</div>

<div align="center">

![PYTHON VERSION](https://img.shields.io/badge/Python-3.8-blue) &nbsp; ![License](https://img.shields.io/badge/GPL-3.0-red) &nbsp; ![Tensorflow Version](https://img.shields.io/badge/Tensorflow-2.7.0-green.svg)

</div>

## About this Repository

This repository is the project repo of ECE1512 2022W at University of Toronto. We make the repository public only on purpose for reviewing and grading. This approach will no longer be maintained after the 2020 Winter term, but releases and updatings will be continuously posted here.

## Repository Structure

This repository will be as two project folder, Project A and B. The structure of Project A is as followed.

```
└── ProjectA
    ├── Project_A_Supp
    │   ├── attruibution methods
    │   ├── hmt_dataset
    │   │   ├── HMT_test
    │   │   │   ├── 01_TUMOR
    │   │   │   ├── 02_STROMA
    │   │   │   ├── 03_COMPLEX
    │   │   │   ├── 04_LYMPHO
    │   │   │   ├── 05_DEBRIS
    │   │   │   ├── 06_MUCOSA
    │   │   │   ├── 07_ADIPOSE
    │   │   │   └── 08_EMPTY
    │   │   └── HMT_train
    │   │       ├── 01_TUMOR
    │   │       ├── 02_STROMA
    │   │       ├── 03_COMPLEX
    │   │       ├── 04_LYMPHO
    │   │       ├── 05_DEBRIS
    │   │       ├── 06_MUCOSA
    │   │       ├── 07_ADIPOSE
    │   │       └── 08_EMPTY
    │   ├── log2
    │   │   └── train
    │   └── models
    └── Report
        └── graphs
```

## Install Dependencies

We provided easy set-up for our notebooks and experiments, but commands are not guaranteed to operate in all the systems. Our test environment is based on Python 3.8/3.9 on Windows 10/11 AMD64.

Install the Python requirements using

```sh
pip install -r requirements.txt --upgrade
```

Recommend to use module mode for PYPI, try this while using Python 3.8 in Windows

```sh
py -3.8 -m pip install -r requirements.txt --upgrade
```
