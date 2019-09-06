
## MLflow Tutorial for Start Conference New York, 2019
### Agenda
 * Introduction
 * MLflow: What, Why, and How
  * MLflow Concepts: 
    * Philosophy & Motivation
    * Tracking, Projects & Models
    * Deployment
  * Road Map 1.x
  * Community and Adoption
  * Get Involved!
  * Managed MLflow Demo
 * Q & A
 * Setting up your environment
 * Hands-on Tutorials
 
### Prerequisites 
1. Knowledge of Python 3 and programming in general
2. Preferably a UNIX-based, fully-charged laptop with 8-16 GB, with a Chrome or Fixfox brower
3. Familiarity with GitHub, git, and account on Github
4. Some Knowledge of some Machine Learning Concepts, Libraries, and Frameworks 
     * scikit-Learn
     * pandas and Numpy
     * Apache Spark MLlib
     * TensorFlow/Keras
5. PyCharm/IntelliJ or Choice of sytnax-highligted Python Editor
6. pip/pip3 and Python 3 installed
7. Loads of laughter, curiosity, and sense of humor ... :-)

### Installation and Setup environment

1. Load MLflow [docs](https://mlflow.org) in your browser
 * Click on Documentation (and keep this tab open)
2. `git clone git@github.com:dmatrix/spark-saturday.git`
3. `cd <your_cloned_directory>/tutorials/mlflow/src/python`
4. Install MLflow and the required Python modules 
    * `pip install -r req.txt` or `pip3 install -r req.txt`
5. `cd labs`
6. If using PyCharm or IntelliJ, create a project and load these files in the project

## Labs 
The general objective of the labs are to create a baseline or a benchmark model,
followed by creating experimental models by tuning parameters to affect a better outcome. 
This is achived by experimenting and tracking the effects, using MLflow tracking
APIs. In simpler terms:

* 1. Train a base line model with initial parameters
* 2. Record the relevant metrics and parameters with MLflow APIs
* 3. Observe the results via MLflow UI
* 4. Change or tweak relevant parameters
* 5. Repeat 2-4 until satisfied

This above iterative process is recurrent in each of the lab. 

### Lab-1: Sckit-Learn Regression with RandomForestRegressor 
 * _petrol_regression_lab_1.py_
 
 Objectives of this Lab: 
 
 * Use RandomForestRegressor Model
 * How to use the MLflow Tracking API
 * Use the MLflow API to experiment several Runs
 * Interpret and observe runs via the MLflow UI
 
#### Lab-1 Excercises: 

 1. Consult [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) documentation 
 2. Change or add parameters, such as depth of the tree or random_state: 42 etc.
 3. Change or alter the range of runs and increments of n_estimators
 4. Check in MLfow UI if the metrics are affected
 
 *challenge-1:* Create mean square error and r2 artifacts and save them for each run
 
 *challenge-2:* Use linear regression model and see if it makes a difference in the evaluation metrics

### Lab-2: Sckit-Learn Classification with RandomForestClassifier
 * _banknote_classification_lab_2.py_
 
Objectives of this lab:
 * Use a RandomForestClassification Model
 * How to use the MLflow Tracking API
 * Use the MLflow API to experiment several runs
 * Interpret and observe runs via the MLflow UI
 
#### Lab-2 Excercises: 
  * Consult [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) documentation
  * Change or add parameters, such as depth of the tree or random_state: 42 etc.
  * Change or alter the range of runs and increments of n_estimators
  * Check in MLfow UI if the metrics are affected
  * Log confusion matrix, recall and F1-score as metrics
  
 [Nice blog to read](https://joshlawman.com/metrics-classification-report-breakdown-precision-recall-f1/),
 [Nice blog to read](https://towardsdatascience.com/understanding-random-forest-58381e0602d2) 
 [source for Lab 1 & 2](https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/), and  
 [data source for lab 1 & 2](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

### Lab-3: Sckit-Learn Regression Base with RandomForestRegressor 
  * _airbnb_base_lab_3.py_

Objectives of this lab:
 * How to use the MLflow Tracking API
 * Interpret and observe runs via the MLflow UI
  
#### Lab-3 Excercises: 
  *  Run script and simple base line model
  *  Observe the parameters and metrics in the MLflow UI
  
### Lab-4: Sckit-Learn Regression Experimental with RandomForestRegressor \
* _airbnb_exp_lab_4.py_
 
Objectives of this lab:
 * Create experiments and log meterics and parameters
 * Interpret and observe runs via the MLflow UI
 * How to use _MLflowClient()_ API to peruse experiment details
 
#### Lab-4 Excercises: 
  * Modify or extend the parameters
  * Compare the results between baseline and experimental runs
  * Did the experimental runs produce better outcomes of metrics?
  * Did the RMSE decrease over the experiments

 [Nice blog to read](https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e)
 
### Lab-5 : Deep Learning Neural Networks for Classification
* _keras_mnist_lab_5.py_

Objectives of this lab:
 * Introduce Keras NN Model
 * Create your own experiment name and log runs under it
 
#### Lab Excercises: 
 * Consult [Keras Sequential Model](https://keras.io/getting-started/sequential-model-guide/) documentation
 * Change or modify Neural Network and regularization parameters
    * Add hidden layers
    * Make hidden units larger
    * Try a different [Keras optimizers](https://keras.io/optimizers/): RMSprop and Adadelta
    * Train for more epochs
 * Log parameters, metrics, and the model
 * Check MLflow UI and compare metrics among different runs

### Lab-6: Loading and predicting an existing model 
* _load_predict__model_lab_6.py_

Objectives of this lab:
 * loading an existing model and predicting with test data
 * using pyfunc model function
 *challenge-1:* Can you load and predict other models from the labs?
  * Lab-3, lab-4 or lab-5?
 
#### Lab Excercises: 
 * Extend the _MLflowOps_ class private instance dictionary of 
 function mappers to include [pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model) model
 * Use a couple of the run_uid from your Lab-1 runs. 
  * Check your MLflow UI for run_uids
 * Use the _load_model_type.predict(test_data)_ to predict tht outcome
 
### Lab-7 (optional): Executing MLproject from GitHub

Objectives of this lab:
 * Understanding MLflow Project files
 * Running MLflow Project as unit of execution
 
#### Lab Excercises:
 * Execute an existing MLproject on git
 * Consult [docs]( https://mlflow.org/docs/latest/quickstart.html#running-mlflow-projects
) for running MLprojects
 * Can you create an MLproject for one of these labs?
 * Can you execute it with different parameters?
 * `mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5`
 * with no-conda use `mlflow run --no-conda https://github.com/mlflow/mlflow-example.git -P alpha=5`

### Lab-8 (Capstone): Experiment your own model of choice

Objectives of this lab:
 * Use and build whatever you have learned from above
 
 #### Lab Excercises:
 * Create a Python script for your example
 * Consult [MLflow](https://mlflow.org/docs/latest/python_api/mlflow.html) and [Tracking APIs](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html) to log:
    * individual or bulk parameters and  metrics
    * add tags or notes
    * experiment different parameters with each run
    * can you create artifacts (mathplot) and save them?
    * Consult MLflow UI to pick the best model
    * Can you load the best model using native model or pyfunc?
    * Can you predict with test data?
    
Some possible choices to use as a baseline model:

1. [Tensorflow/Keras MNIST fashion Model](https://www.tensorflow.org/tutorials/keras/basic_classification)
2. [MLflow Examples from Github repo](https://github.com/mlflow/mlflow/tree/master/examples)
3. [Deep Learning with Python: Francois Challet](https://github.com/fchollet/deep-learning-with-python-notebooks)

Let's Code and Have Loads of Fun!

Cheers

Jules
