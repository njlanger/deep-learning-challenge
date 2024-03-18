Overview:

The purpose of this project was to create and evaluate a deep learning model using TensorFlow and keras for Alphabet Soup, a fictitious charitable organization. The goal was to predict whether applicants would be successful if funded based on various features such as application type, affiliation, classification, organization type, income amount, and fundraising ask amount. After the initial model set-up, training and evaluation, the request was to optimize the model for an accuracy level of 75% or greater. 

Results:

The first attempt of the model resulted in an accuracy level of 72.58%.

    268/268 - 1s - loss: 0.5527 - accuracy: 0.7258 - 650ms/epoch - 2ms/step
    Loss: 0.5526813864707947, Accuracy: 0.7258309125900269

After working to optimize the model, the accuracy level reached 75.01%, which indicates reasonable performance.

    204/204 - 0s - loss: 0.5324 - accuracy: 0.7502 - 482ms/epoch - 2ms/step
    Loss: 0.5324048399925232, Accuracy: 0.7501533031463623

Data Preprocessing:
 
The target variable that was used for the model was the column (variable) ‘IS_SUCCESSFUL’. This represented whether the funding application was successful (1) or not (0).
The features used to train the model were all the other columns (variables).
In the initial training of the model columns (variables) that were removed were ‘EIN’, and ‘NAME’. In the optimization of the model, ‘SPECIAL_CONSIDERATIONS’, and ‘USE_CASE’ columns (variables) were also removed.

Compiling, Training, and Evaluating the Model:

Durning the initial model set-up, two hidden layers with 8 and 5 neurons respectively were used. The model was trained using 100 epochs. During the optimization of the model a third hidden layer was used with 12, 6, and 16 neurons respectively and 50 epochs used for training. During both the initial and optimization the hidden layers utilized ReLU activation functions. The output layer utilized a sigmoid activation function because of the problems’ binary classification. The number of neurons and layers was chosen based on experimentation, and what was learned in class, balancing model complexity with computational resources.
	
  Steps to increase model performance: (these are also discussed above)	
  
    •Deleted two additional columns (variables), ‘SPECIAL_CONSIDERATIONS’, and ‘USE_CASE’
    •Added another hidden layer
    •Adjusted the number of neurons
    •Reduced the number of training epochs
    
Summary:

The deep learning model achieved a decent accuracy of just over 75% after optimization.  However, there is room for improvement. To further enhance the model performance, several strategies can be considered:
  -	Feature Engineering: Explore additional features or create new features from existing ones that might provide more predictive power.
  -	Hyperparameter Tuning: Conduct a more extensive hyperparameter search using techniques like grid search or random search to find optimal values for parameters such as learning rate, batch size, and number of epochs.
  -	Model Architecture: Experiment with different neural network architectures, including varying the number of layers, neurons per layer, and activation functions.
A different model, or a neural network with a more complex architecture like a Recurrent Neural Network, could potentially provide better performance by capturing more intricate patterns and relationships in the data. The choice of model should be guided by the specific characteristics of the dataset and the computational resources available.

This assignment was done from what was learned in class, ChatGPT was used as a reference for model optimization. 
