# deep-learning-challenge

# Requirement

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With the knowledge of machine learning and neural networks, we have to use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, we have a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

# Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, we have to preprocess the dataset. This step prepares for Step 2, where we have to compile, train, and evaluate the neural network model.

The following instructions for preprocessing steps are to be completed:
1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
  a. What variable(s) are the target(s) for your model?
  b. What variable(s) are the feature(s) for your model?
2. Drop the `EIN` and `NAME` columns.
3. Determine the number of unique values for each column.
4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
6. Use `pd.get_dummies()` to encode categorical variables.
7. Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.
8. Scale the training and testing features datasets by creating a `StandardScaler instance`, fitting it to the training data, then using the `transform` function.

# Step 2: Compile, Train, and Evaluate the Model
Using the knowledge of TensorFlow, we have to design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. we need to need to think about how many inputs there are before determining the number of neurons and layers in your model. Once that step has been completeded, we have to compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.
1. Continue using the file in Google Colab in which the preprocessing steps from Step 1 were performed.
2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every five epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

# Step 3: Optimize the Model
Using the knowledge of TensorFlow, optimize the model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize your model:
1. Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  a. Dropping more or fewer columns.
  b. Creating more bins for rare occurrences in columns.
  c. Increasing or decreasing the number of values for each bin.
  d. Add more neurons to a hidden layer.
  e. Add more hidden layers.
  f. Use different activation functions for the hidden layers.
  g. Add or reduce the number of epochs to the training regimen.

Note: 
1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
2. Import the dependencies and read in the charity_data.csv to a Pandas DataFrame.
3. Preprocess the dataset as in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export the results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

# Step 4: Write a Report on the Neural Network Model
For this part of the assignment, we have to write a report on the performance of the deep learning model you created for Alphabet Soup. The report should contain the following:
1. <b> Overview of the analysis:</b> <br> With the knowledge of machine learning and neural networks, we have to use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
2. <b>Results:</b> Using bulleted lists and images to support your answers, address the following questions: <br>
  a. Data Preprocessing <br>
    i.  What variable(s) are the target(s) for your model? - The target variable for the model is `IS_SUCCESSFUL`, which indicates whether the applicant was successful (1) or not sucessful (0) in receiving funding. <br>
    ii. What variable(s) are the features for your model? - The feature variables for the model are as follows: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT` <br>
    iii.What variable(s) should be removed from the input data because they are neither targets nor features? - The  variable removed from the input data are: `EIN` and `NAME` in the inital model. I have removed two other varibales in the optimization models, which are: `STATUS` ans `SPECIAL_CONSIDERATIONS`. <br>
  b. Compiling, Training, and Evaluating the Model <br>
    i.  How many neurons, layers, and activation functions did you select for your neural network model, and why? - I used <b>two</b> hidden layers in all the models. In the inital model, I used 9 neurons with `relu` activation function in the first hidden layer and 5 neurons with `relu` activation function in the second hidden layer, and `sigmoid` activation function in the output layer. In the first optimization model, I used 100 neurons with `relu` activation function in the first hidden layer and 100 neurons with `relu` activation function in the second hidden layer, and `sigmoid` activation function in the output layer. In the second optimization model, I used 100 neurons with `relu` activation function in the first hidden layer and 100 neurons with `relu` activation function in the second hidden layer, and `sigmoid` activation function in the output layer.<br>
    ii. Were you able to achieve the target model performance? <br>
    iii.What steps did you take in your attempts to increase model performance? <br>
3. <b>Summary:</b> Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation. - <br>

# References
Referred to various class activity exercises, got support from Assistant Instructor, and websites: https://scikit-learn.org.

# Files submitted including this README File
-> credit-risk-classification Folder <br>
a. Resources Folder -> lending_data.csv (contains the CSV file) <br>
b. credit_risk_classification.ipynb (contains the srcipt) <br>
c. report-template.md (Analysis Report is written using this report template)
