

Here are the explanation of the what it is included in the Supplementary file:


    1. mnist1d_utils.py is a python file to re-create the MNIST-1D dataset. (Based on: https://github.com/greydanus/mnist1d)

    2. MNIST1D.pkl is a “pickle file” containing all data in the MNIST-1D dataset. The data in this file is save as a dictionary that can be also created using the function make_dataset() in the library mnist1d_utils.py. More information to read the dictionary is provided in the notebook MNIST1D.ipynb.
 
    3. hmt_dataset is a folder containing two subfolders, including the train and test set for the HMT dataset.

    4. xai_utils.py is a Python file including utility functions needed for three state-of-the-art solutions in the field of visual XAI, Grad-CAM, RISE (Randomized Input Sampling for Explanation), and SISE (Semantic Input Sampling for Explanation). 

    5. HMT.ipynb and MNIST1D.ipynb are two Python notebooks showing 1) how the models are trained with each of the two described datasets, and 2) how the explanation algorithms included in xai_utils.py are applied on each of the described models.

    6. models is a folder containing the pre-trained VGG7 model for HMT dataset.

    7. Project A_FAQs: A list of Frequently Asked Questions which try to throw light on (almost) all of your questions and concerns which you may have during the course of Project A.


