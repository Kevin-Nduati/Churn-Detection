# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The data contains information about bank customers such as job, age, marital info, education, whether or not they had housing. We try to predict whether 
a customer will subscribed a term deposit.
The data was first fitted to a Logistic Regression model. The model managed an accuracy of 91.5%. After this, I ran an AutoMl, where the data was fitted to 40 models. The best performing model was a VotingEnsemble with an accuracy of 91.44%, followed by MaxAbsScaler XGBoostClassifier with an accuracy of 91.2%.

## Scikit-learn Pipeline
### The Pipeline Architecture
* Create a tabular dataset using TabularDatasetFactory. The data is from <a href = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv">Dataset</a>
* train python scriptcontains data that is used to clean the data, and enode categorical values.
* Data is then split into train and test sets
* A Logistic Regression model is then fitted to the data
* I then used hyperdrive so as to tune hyperparameters(C and max_iter).

The results of the hyperdrive are as shown below:<br>
<img src="https://github.com/Kevin-Nduati/Udacity-Project/blob/df9095aa232b556673ba7afde258016243336856/images/hyperdrive.png">

### Benefits of Random Parameter Sampler
Random Sampling was chosen as hyperparameter values were selected from the defined search space. inverse regularization strength discrete values were 0.01, 5, 20, 100, 500. Lower values indicate strong regularization. As for max iteration were 10,50, 100,15,200.

### Benefits of Bandit Policy for Early Stopping
The early stopping policies automatically terminates poorly performing runs. I defined a slack factor of 0.1, where the policy terminates runs where the primary metric, in this case accuracy, is not within the specified slack factor compared to the best performing run

## AutoML
We configured the AutoML with the following parameters:
* **task** - whether it is a classification or regression problem. In this case, we chose classification.
* **primary_metric** - This is the metric that we want the autoML to tune. In this case, we prioritize accuracy
* **training_data** - specify the data to be used during training
* **label_column_name** - label of the column that will be predicted
* **n_cross_validations** - number of cross validations that were performed. In this case, I chose 3

These are the results of the AutoML:
<img src="https://github.com/Kevin-Nduati/Udacity-Project/blob/master/images/automl.png">
<br><br>
The best model is:<br>
<img src="https://github.com/Kevin-Nduati/Udacity-Project/blob/master/images/model.png">

## Pipeline comparison
The Logistic Regression model tuned with hyperdrive had an accuracy of 91.5% while the VotingEnsemble from AutoMl had an accuracy of 91.44%. This means that there was no significant difference in the two models 


## Future work
The data was highly imbalanced. I would like to perform balancing techniques and see whether there will be improvements in the model.

## Proof of cluster clean up
<img src="https://github.com/Kevin-Nduati/Udacity-Project/blob/master/images/clean.png">
