# Wine Rating Predictor

In this procject, I built a wine rating predictor for an online wine seller. This wine predictor aims to show good prediction is possible using the ```wine_dataset``` . 

Wine rating is a score between 80 and 100 and represents the quality of wine. With the current set of features, random forest classifier and its tuned parameter wine rating predictor can predict the quality of wine with the mean square error of 4.9. This metric shows that fully-automated machine learning solution on production is feasible and effective for the client.

This predictor runs the machine learning pipeline with Docker and Luigi tasks. So, it can be run on any machine that has docker and docker-compose installed.

Machine learning pipeline consists of the steps below:

1. Download Data
2. Make Dataset
3. Clean Data
4. Extract Features
5. Transform Data
6. Impute Data
7. Train Model
8. Evaluate Model

and creates a file contains the random forest model, and evaluation plots of the model performance.

The output file of the step 1 can be found in the data_root > raw. The output files of the steps 2, 3, 4, 5 and 6 can be found in the data_root > interim. The final output files of step 7 and 8 can be found in the data_root > output.

The exploration of the wine_dataset and the chain of thougts of for the feature and model selection can be found in the notebooks in the notebooks folder. The complete machine learning workflow followed in the notebooks are as follows:

1. Understand & Clean & Format Data
2. Exploratory Data Analysis
3. Feature Engineering & Pre-processing
4. Set Evaluation Metric & Establish Baseline
5. Model Selection & Tune Hyperparameters of the Model
6. Train Model
7. Evaluate Model on Test Data
8. Interpret Model Predictions
9. Conclusions

This project is built as part of the interviews for the Machine Learning Engineer position at Data Revenue.
