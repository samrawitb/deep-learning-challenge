# Alphabet Soup Charity Success Prediction - Neural Network Model Report
## Overview of the Analysis
The objective of this analysis is to develop a deep learning model that can predict whether a charity's fundraising campaign will be successful. By analyzing historical data from Alphabet Soup’s database, the neural network model will help identify the key factors contributing to campaign success, enabling the charity to optimize future fundraising efforts.
________________________________________
## Results
### Data Preprocessing
#### •	Target Variable:
The target variable for this model is IS_SUCCESSFUL, which indicates whether a charity campaign was successful or not (binary classification: 1 for successful, 0 for unsuccessful).

### •	Feature Variables:
The features for the model include all columns except the target and irrelevant ID-type columns. The most important features include:

- APPLICATION_TYPE
- CLASSIFICATION
-	USE_CASE
-	ORGANIZATION
-	STATUS
-	INCOME_AMT
-	ASK_AMT

These categorical variables were converted into numeric form using one-hot encoding.

### •	Removed Variables:
The following variables were removed because they are neither features nor targets:
-	##### EIN: This is a unique identifier for organizations and has no predictive value for the model.
-	##### NAME: This column, representing the organization's name, is also irrelevant for predicting the success of the campaign.

## Compiling, Training, and Evaluating the Model

### Model Architecture:
- #### Neurons:
  - First hidden layer: 80 neurons
  - Second hidden layer: 30 neurons
  
- #### Layers:
  -	Input layer (based on the number of features)
  -	Two hidden layers
  -	Output layer (1 neuron for binary classification)
  
- ### Activation Functions:
  -	ReLU (Rectified Linear Unit) was chosen for both hidden layers because it is a popular activation function for neural networks that helps mitigate the vanishing gradient problem.
  -	Sigmoid was used for the output layer to output probabilities, making it suitable for binary classification.

### Model Optimization Attempts
In order to improve the model’s performance and prevent overfitting, I made three different optimization attempts:

### 1. First Optimization: Adding Dropout Layers
- Changes Made:
  - I added dropout layers after each hidden layer to randomly set 20% of the neurons inactive during each training step, helping to prevent overfitting.
  - The first hidden layer was set to have 80 neurons with a ReLU activation function.
  - The second hidden layer had 30 neurons, also using ReLU, followed by a dropout layer.
  - The output layer had 1 neuron with a sigmoid activation for binary classification.

![image](https://github.com/user-attachments/assets/59c64283-9c84-45e8-9807-e3e919d6cbe1)
    
  Outcome: This resulted in a total of 5,981 parameters. While the model performed better with regularization, further optimization attempts were made to see if accuracy could be improved further.


### 2. Second Optimization: Adjusting the Number of Neurons in Hidden Layers
- Changes Made:
  - I reduced the number of neurons in the first hidden layer from 80 to 50 to see if a smaller network with fewer parameters would generalize better.
    
  Outcome: Reducing the number of neurons decreased the overall model size and training time, but the performance on both the training and test sets slightly decreased, indicating that the model was not learning enough with fewer neurons.

### 3. Third Optimization: Adding Another Hidden Layer
- Changes Made:
  - I added a third hidden layer with 20 neurons using ReLU activation. The idea was to introduce more depth to the model and potentially capture more complex patterns.
    
Outcome: Adding the third hidden layer led to a slight improvement in accuracy, but the model also took longer to train. The results indicated that the deeper network was learning more complex relationships, though performance gains were marginal with the added complexity.

  
### Why these choices?:
  - The number of neurons and layers were selected to balance model complexity without overfitting, based on the size of the input features. ReLU is a well-established choice for hidden layers, and Sigmoid is appropriate for binary output tasks.
  
### Model Performance:
  - Accuracy: The final model achieved an accuracy of 72.39% on the test dataset.
  - Loss: The binary cross-entropy loss was 0.5568.
    
While this is a reasonable accuracy, it falls short of ideal predictive performance, leaving room for improvement.

### Steps to Improve Performance:
  - I attempted different combinations of neurons in the hidden layers and increased the number of epochs to allow the model to learn more thoroughly from the data.
  - Feature scaling and one-hot encoding were used to ensure all features contributed equally to the model.
  - Regularization techniques such as dropout could be implemented in future attempts to reduce potential overfitting, though no significant overfitting was observed in this case.
________________________________________
## Summary

### Overall Results

The deep neural network model demonstrated a reasonable performance with an accuracy of 72.39%. The model was able to identify patterns in the dataset and provide a baseline for predicting the success of charity campaigns. However, the model’s performance could be enhanced with additional tuning or by trying different algorithms.

### Recommendation for a Different Model
Given the moderate performance of the neural network, I recommend considering a Random Forest Classifier for this classification problem. Random Forests are ensemble models that can often outperform neural networks on tabular data. They require less preprocessing and can provide insights into feature importance, allowing us to determine which features most impact the success of a charity campaign.

### Why Random Forest?:
  - Handling Categorical Data: Random Forests can handle categorical variables without needing extensive one-hot encoding
  - Feature Importance: They provide easy-to-interpret feature importance scores, which can help us refine the model by identifying the most significant features.
  - Less Tuning Required: Unlike neural networks, Random Forests do not require extensive tuning of hyperparameters like neurons, layers, or activation functions.
    
By switching to Random Forest, we may achieve better accuracy, interpretability, and ease of deployment for predicting charity campaign success.

