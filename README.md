# Explainable Active Learning for Improved Text Classification
 
Active Learning facilitates training of models without requiring large amounts of labelled data alleviating the need of annotating excessive unlabelled data thus avoiding the costs associated with referring to an expert to annotate the data. Among other active learning strategies for selecting a set of samples for training from unlabelled samples, the method using uncertainty sampling has generally been the most popular one, however, it alone may not always be useful in deciding whether the sample to be selected is the most informative. [Another strategy](https://github.com/Ishani-Mondal/Explainable_Active_Learning) which employs uncertainty sampling along with model's interpretability as guiding heuristics has proven to be more effective in improving model's overall performance on image data. This is an implementation of aformentioned strategy on text data, specifically, [IMDb movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/).


### Contents
1. **AL_TextClassification.py** - Main python implementation of the proposed strategy.
2. **AL_TextClassification.ipynb** - A Jupyter notebook interactive version of python implementation, facilitates additional functionality to visualize explanations.
3. **Annotated Dataset** - A human annotated dataset consisting of 50 samples, 25 positive and 25 negative. It serves as ground truth in model evaluation.
4. **requirements.txt** - Python dependencies file.
5. **Results** - Model performance results as obtained during experiments.

### Environment Setup
Before running the python script, all the necessary dependencies would need to be downloaded, in order to do that use:

`pip install -r requirements.txt`

If using conda, create a new virtual environment (replace \<env\> with environment name):
 
`conda create --name <env> --file requirements.txt`
 
 ### Run Script
 After the environment has been successfully setup, use following command to run the script:
 
 `python AL_TextClassification.py [FLAGS]`
 
 Supported Flags-
 - -a : ALEX Method (Proposed Strategy)
 - -r : Random Sampling Method
 - -l : Uncertainty Sampling Least Confidence Method
 - -m : Uncertainty Sampling Smallest Margin Method
 - -A : Run all methods
 
 Example usage:
 
 `python AL_TextClassification.py -a -r`
 
 Executing above command would run ALEX Method and Random Sampling Method.
 
 ### License
 
 This repository has been made available to the public to encourage open research and can be used, reused or modified for non-commercial purposes without prior notice. Please cite this and [ALEX](https://github.com/Ishani-Mondal/Explainable_Active_Learning) repository when using.
