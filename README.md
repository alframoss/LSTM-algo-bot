## LSTM-algo-bot

This GitHub repository contains all the work related to my construction of a Brent crude algorithmic trading strategy using predictions from a deep learning model. Instructions on how to run the program are at the bottom.

## Data

Data is sourced from Yahoo finance. We collect 10 years of daily information for a multitude of features varying from Brent crude price to EUR/USD price to US 10y Treasury yields. It has been constructed such that the user can add/remove features easily by amending the list `FEATURES`. We use a training split of 80%, a cross-validation split of 10% and testing split of 10%. This can also be adjusted by the user by amending `TRAIN_SPLIT`, `VAL_SPLIT` and `TEST_SPLIT`.

## Explanation

### Architecture Choice

The goal of the deep learning model is to predict the subsequent closing price given some features. We decide to use a recurrent neural network (RNN) in the form of an LSTM as time series data suits the sequence aspect of input data for RNN models. I was also curious to see how well it performs. A convolutional neural network (CNN) would also be an interesting choice of architecture. 

### Data Collection & Processing

For each feature, we collect 10 years worth of daily data from Yahoo Finance. We then create a look-back window (construct sequences which is essential for RNNs). For each target close price, find the close price for the previous day, for the day before that and repeat this until a sequence of desired length is formed. This is done for each feature collected. I choose to look back to the three last days with a variable called `SEQUENCE_LENGTH` but this can also be adjusted to the users choice.

### Training, Cross-Validation & Testing

We then split the sequences using a a training split of 80%, a cross-validation split of 10% and testing split of 10%. This can also be adjusted by the user by amending `TRAIN_SPLIT`, `VAL_SPLIT` and `TEST_SPLIT`.

### Feature Scaling

Using scikit-learn's `MinMaxScaler`, I scale the training data between 0 and 1. This helps to improve the time spent training the model. We use the same scale to scale the cross-validation and testing data. It is important to scale after splitting the data to prevent any leakage of information to the cross-validation and testing data.

### Training Model

We then train the model using the TensorFlow library. A sequential model is setup that uses a Gaussian noise layer to reduce overfitting, followed by an LSTM layer with 64 neurons and a dense layer with 1 neuron. The number of neurons in the LSTM layer can be varied to the users liking. I choose to use 50 Epochs and a batch size of 1 which may slow the training time, however I prioritise accuracy. We use a mean-squared-error loss function and have a variable learning rate using the `adam` optimizer. More Epochs can be used for more accurate results.

### Regularisation

I use L1 (LASSO) regularisation to the weights which aims to reduce weights by adding a penalty for the absolute value of weights. Dropouts and early stopping can also be used.

### Algorithm

Using this prediction, I construct a mean-reverting trade strategy. If we predict that the price of Brent will increase, we long. If we predict that the price of Brent will decrease, we short. The size of the postion depends on how far the predicted price deviates from the current open price of Brent.

### Transaction Costs

While this works much better without transaction costs, we would like to see how well this algo would perform in an actual market. For that sake, we use artificial transaction costs. We set a bid-ask spread equal to the standard deviation of the open Brent prices. For those who do not know what the bid-ask spread is, I recommend checking https://www.investopedia.com/terms/b/bid-askspread.asp#:~:text=A%20bid%2Dask%20spread%20is,seller%20is%20willing%20to%20accept.

## Results

### Predictions

![Algo_Predictions](https://github.com/user-attachments/assets/18a99618-cfe6-452b-bf1e-3848be8ea99f)

![Algo_Test_Predictions](https://github.com/user-attachments/assets/3afa2e67-f1bb-4708-b4cf-edb1e12a1680)

### Accuracy Metrics

- Mean Absolute Error (MAE) : 1.68
- Mean Squared Error (MSE) : 4.15
- Root Mean Squared Error (RMSE) : 2.04
- $R^2$ : 0.82

### Algorithmic Strategy

We outperform the buy and hold strategy.

![Algo_Signals](https://github.com/user-attachments/assets/6e655b6b-2554-4384-94f2-fc381ec2e10d)

![Algo_PnL](https://github.com/user-attachments/assets/7f803f85-c653-4516-aaad-19de1443b42d)

## Instructions

1. Setup a virtual environment

`python3 -m venv venv-python`

2. Activate the virtual environment

`source venv-python/bin/activate`

3. Install necssary python libraries to run the program

`pip3 install -r requirements.txt`

4. Run the program

`python3 18082024_LSTM_Brent_Algo.py`
