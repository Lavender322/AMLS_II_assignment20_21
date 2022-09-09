# AMLS_II_assignment20_21

### Description

A brief description of the organization of the project is presented below in a logical order: 

1. Step 1 - **Data Loading**: data.py
1. Step 2 - **Data Pre-processing**: data.py
1. Step 3 - **Training and Validating with Specified Model**: train.py, model/edsr.py, model/srgan.py, model/shared.py
1. Step 4 - **Evaluation on Test Set**: utils.py, model/shared.py
1. Step 5 - **Project Execution**: main.py

### Usage

The role of each file in this project is illustrated as follows:

* The main.py script contains the main body of this project, which is run only to train, validate, and test the optimal machine learning model selected for the specified four tasks. 
* The **data.py** module defines a data loader to automatically download DIV2K to disk and performs a series of pre-processing steps.
* The **utils.py** module defines utility functions.
* The **train.py** module defines training precedures for all the models.
* The **model/shared.py** module defines shared functions of the neural network operator (namely pixel shuffle), normalisation, evaluation metrics.
* The **model/edsr.py** model defines the original EDSR model network architecture, and a modified EDSR network architecture with weight normalisation performed.
* The **model/srgan.py** module defines the original SRGAN model network architecture.
* The **main.py** script executes the program with preset settings and network architecture.
* The **demo** folder contains three x4 LR image patches from DIV2K validation dataset and the Set 5 for the purpose of demonstration.
