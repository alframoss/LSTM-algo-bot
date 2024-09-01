import numpy as np
import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras import layers, Sequential, Input

START_DATE = dt.datetime.now() - dt.timedelta(days=365 * 10)
END_DATE = dt.datetime.now()

# Given a list of features, collect market data using Yahoo Finance function.

FEATURES = [
    ("BZ=F", ["Open", "Adj Close", "Volume"]),
    ("CL=F", ["Open", "Adj Close", "Volume"]),
    ("NG=F", ["Open", "Adj Close", "Volume"]),
    ("^TNX", ["Open", "Adj Close"]),
    ("EURUSD=X", ["Open", "Adj Close"]),
    ("CNY=X", ["Open", "Adj Close"]),
]

# Construct sequences for LSTM input.

SEQUENCE_LENGTH = 3

df = pd.DataFrame()
for asset, features in FEATURES:
    fetched_df = yf.download(asset, start=START_DATE, end=END_DATE)[features]
    for i in features:
        for j in range(SEQUENCE_LENGTH - 1, -1, -1):
            df[f"{i} {asset} - {j}"] = fetched_df[i].shift(j)

# The aim of the neural network is to predict the next price, this the target.

df["Target"] = df["Adj Close BZ=F - 0"].shift(-1)
df = df.dropna()

noRows = df.shape[0]
noFeats = int((df.shape[1] - 1) / SEQUENCE_LENGTH)  # Excluding target column.

# Split the DataFrame containing features into training, cross-validation and testing data.

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

train = df.drop("Target", axis=1).iloc[0 : int(noRows * TRAIN_SPLIT), :]
y_train = np.array(df["Target"].iloc[0 : int(noRows * TRAIN_SPLIT)])
val = df.drop("Target", axis=1).iloc[int(noRows * TRAIN_SPLIT) : int(noRows * (TRAIN_SPLIT + VAL_SPLIT)), :]
y_val = np.array(df["Target"][int(noRows * TRAIN_SPLIT) : int(noRows * (TRAIN_SPLIT + VAL_SPLIT))])
test = df.drop("Target", axis=1).iloc[int(noRows * (TRAIN_SPLIT + VAL_SPLIT)) :, :]
y_test = np.array(df["Target"][int(noRows * (TRAIN_SPLIT + VAL_SPLIT)) :])

# Conduct feature scaling using scikit-learn.

scaler = MinMaxScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

# Reshape into a suitable form for LSTM (# of samples, # of sequences, # of features).

X_train = np.zeros([train.shape[0], SEQUENCE_LENGTH, noFeats])
X_val = np.zeros([val.shape[0], SEQUENCE_LENGTH, noFeats])
X_test = np.zeros([test.shape[0], SEQUENCE_LENGTH, noFeats])

for i in range(train.shape[0]):
    for j in range(SEQUENCE_LENGTH):
        for k in range(noFeats):
            X_train[i, j, k] = train[i, k * SEQUENCE_LENGTH + j]

for i in range(val.shape[0]):
    for j in range(SEQUENCE_LENGTH):
        for k in range(noFeats):
            X_val[i, j, k] = val[i, j + k * SEQUENCE_LENGTH]

for i in range(test.shape[0]):
    for j in range(SEQUENCE_LENGTH):
        for k in range(noFeats):
            X_test[i, j, k] = test[i, j + k * SEQUENCE_LENGTH]

# Construct neural network using LSTM architecture.

LSTM_NEURONS = 64
EPOCHS = 50

std = df["Adj Close BZ=F - 0"].pct_change().dropna().std()

model = Sequential(
    [
        Input(shape=(SEQUENCE_LENGTH, noFeats)),
        layers.GaussianNoise(std),
        layers.LSTM(units=LSTM_NEURONS),
        layers.Dense(units=1),
    ]
)

model.summary()
model.compile(loss="mean_squared_error", optimizer="adamax")
h = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=1)
y_test_hat = model.predict(X_test)

# Plot predictions and model loss.

csfont = {"fontname": "Helvetica"}
blmbg_black = "#000000"
blmbg_or = "#fb8b1e"
blmbg_blue = "#0068ff"
blmbg_red = "#ff433d"
blmbg_cyan = "#4af6c3"
plt.style.use("classic")

y_train_hat = model.predict(X_train)
y_val_hat = model.predict(X_val)

plt.figure(figsize=(15, 6))
plt.plot(
    df.index, df["Target"], label="True Close Price", color=blmbg_black, linewidth=1.3
)
plt.plot(
    df.iloc[0 : int(noRows * TRAIN_SPLIT), :].index,
    y_train_hat,
    label="Training Predictions",
    color=blmbg_blue,
    linewidth=1.3,
)
plt.plot(
    df.iloc[
        int(noRows * TRAIN_SPLIT) : int(noRows * (TRAIN_SPLIT + VAL_SPLIT)), :
    ].index,
    y_val_hat,
    label="Cross-Validation Predictions",
    color=blmbg_red,
    linewidth=1.3,
)
plt.plot(
    df.iloc[int(noRows * (TRAIN_SPLIT + VAL_SPLIT)) :, :].index,
    y_test_hat,
    label="Testing Predictions",
    color=blmbg_or,
    linewidth=1.3,
)
plt.title("LSTM Brent Crude Price Prediction")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend(loc="upper left")
# plt.savefig("Algo_Predictions.png", dpi=400)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].plot(y_test_hat, label="Predicted Close Price", color=blmbg_or, linewidth=1.3)
ax[0].plot(y_test, label="True Close Price", color=blmbg_black, linewidth=1.3)
ax[0].set_title("LSTM Brent Crude Price Prediction")
ax[0].set_xlabel("Day")
ax[0].set_ylabel("Price")
ax[0].legend()
ax[1].plot(h.history["loss"], label="Training Loss", color=blmbg_blue, linewidth=1.3)
ax[1].plot(
    h.history["val_loss"], label="Validation Loss", color=blmbg_red, linewidth=1.3
)
ax[1].set_title("LSTM Model Loss")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend()
# plt.savefig("Algo_Test_Predictions.png", dpi=400)
plt.show()

# Compute accuracy metrics.

mae = mean_absolute_error(y_test, y_test_hat)
mse = mean_squared_error(y_test, y_test_hat)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_hat)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Constructing a mean-reversion algorithm.

buy = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])
sell = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])
buy_exit = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])
sell_exit = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])
buy_small = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])
buy_big = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])
sell_small = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])
sell_big = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])
buy_plot = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])
sell_plot = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])

predicted_price = y_test_hat[:, 0]
actual_price = df["Open BZ=F - 0"].iloc[int(noRows * (TRAIN_SPLIT + VAL_SPLIT)) + 1 :].values
exit_price = df["Open BZ=F - 0"].iloc[int(noRows * (TRAIN_SPLIT + VAL_SPLIT)) + 1 :].shift(-1).values

buyhold_strat = np.zeros([int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1])

# Using an artificial transaction cost based on the standard deviation of the Brent open price.

BASPREAD = df["Open BZ=F - 0"].pct_change().dropna().std()
S1, S2 = 10, 50

for i in range(int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1):
    pred = predicted_price[i]
    curr = actual_price[i]
    exit = exit_price[i]
    if pred > curr:
        if pred > (1 + BASPREAD)**1.2 * curr:
            buy[i] = S2 * curr + BASPREAD
            buy_exit[i] = S2 * exit 
            buy_plot[i] = curr
            buy_big[i] = curr
        elif pred > (1 + BASPREAD) * curr:
            buy[i] = S1 * curr + BASPREAD
            buy_exit[i] = S1 * exit
            buy_plot[i] = curr
            buy_small[i] = curr
    else:
        if pred < (1 - BASPREAD)**1.2 * curr:
            sell[i] = S2 * curr
            sell_exit[i] = S2 * exit + BASPREAD
            sell_plot[i] = curr
            sell_big[i] = curr
        elif pred < (1 - BASPREAD) * curr:
            sell[i] = S1 * curr
            sell_exit[i] = S1 * exit + BASPREAD
            sell_plot[i] = curr
            sell_small[i] = curr
    buyhold_strat[i] = (-actual_price[0]+actual_price[i]) * 100

# Compute alpha generated and plot illustrative graphs.

pnl = (-np.cumsum(buy) + np.cumsum(buy_exit) + np.cumsum(sell) - np.cumsum(sell_exit)) * 10

plt.figure(figsize=(15, 6))
plt.plot(actual_price, label="True Open Price", color=blmbg_black, linewidth=1.3)
plt.plot(y_test_hat, label="Predicted Close Price", color=blmbg_or, linewidth=1.3)
plt.plot(
    buy_plot,
    color="g",
    marker="^",
    linestyle="None",
    label="Buy $p > (1+\\sigma)a$",
)
plt.plot(
    sell_plot,
    color="r",
    marker="v",
    linestyle="None",
    label="Sell $p < (1-\\sigma)a$",
)
plt.title("LSTM Mean Reversion Algorithm")
plt.xlabel("Day")
plt.ylabel("Price")
plt.ylim(0.8 * np.min(actual_price), 1.2 * np.max(actual_price))
plt.legend()
# plt.savefig("Algo_Signals.png", dpi=400)
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(pnl, label="Mean Reversion Algorithm", color=blmbg_black, linewidth=2)
plt.plot(buyhold_strat, label="Buy & Hold Strategy", color=blmbg_or, linewidth=2)
plt.title("LSTM Mean Reversion Algorithm PnL")
plt.xlabel("Day")
plt.ylabel("PnL")
plt.legend(loc="upper left")
# plt.savefig("Algo_PnL.png", dpi=400)
plt.show()
