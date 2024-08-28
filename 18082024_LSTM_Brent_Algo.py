import numpy as np
import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ideas List:
# Things like moving averages could be included
# More features related to brent and energy
# LASSO regression?
# Dropout layers? - Expensive and more training time
# Multiple LSTM layers? - Expensive and more training time
# Hyperparamter tuning using optuna for number of neurons in LSTM

START_DATE = dt.datetime.now() - dt.timedelta(days=365 * 10)
END_DATE = dt.datetime.now()

# Given a list of features, collect market data using Yahoo Finance function.

FEATURES = ["BZ=F", "CL=F", "NG=F", "^TNX", "EURUSD=X", "CNY=X"]

df = pd.DataFrame()
for feature in FEATURES:
    fetched_df = (
        yf.download(feature, start=START_DATE, end=END_DATE)
        .drop(["Close", "Volume", "Low", "High"], axis=1)
        .add_suffix(" " + feature)
    )
    df = pd.concat([df, fetched_df], axis=1)

# The aim of the neural network is to predict the next price, this the target.

df["Target"] = df["Adj Close BZ=F"].shift(-1)
df = df.dropna()

# Split the DataFrame containing features into training, cross-validation and testing data.

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

noRows = df.shape[0]

train = df.iloc[0 : int(noRows * TRAIN_SPLIT), :]
val = df.iloc[int(noRows * TRAIN_SPLIT) : int(noRows * (TRAIN_SPLIT + VAL_SPLIT)), :]
test = df.iloc[int(noRows * (TRAIN_SPLIT + VAL_SPLIT)) :, :]

X_train, y_train = train.drop("Target", axis=1).values, train["Target"].values
X_val, y_val = val.drop("Target", axis=1).values, val["Target"].values
X_test, y_test = test.drop("Target", axis=1).values, test["Target"].values

# Feature scaling using scikit-learn

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape into a suitable form for LSTM (# of samples, # of sequences, # of features).

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Construct neural network using LSTM architecture.

LSTM_NEURONS = 64
EPOCHS = 50

std = df["Adj Close BZ=F"].pct_change().dropna().std()

model = Sequential(
    [
        Input(shape=(1, X_train.shape[2])),
        layers.GaussianNoise(std),
        layers.LSTM(units=LSTM_NEURONS),
        layers.Dense(units=1),
    ]
)

model.summary()
model.compile(loss="mean_squared_error", optimizer="adamax")
h = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=1)
y_test_hat = model.predict(X_test)

# Plot predictions and model loss

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
plt.plot(df.index, df["Target"], label="True Close Price", color=blmbg_black, linewidth=1.3)
plt.plot(df.iloc[0 : int(noRows * TRAIN_SPLIT), :].index, y_train_hat, label="Training Predictions", color=blmbg_blue, linewidth=1.3)
plt.plot(df.iloc[int(noRows * TRAIN_SPLIT) : int(noRows * (TRAIN_SPLIT + VAL_SPLIT)), :].index, y_val_hat, label="Cross-Validation Predictions", color=blmbg_red, linewidth=1.3)
plt.plot(df.iloc[int(noRows * (TRAIN_SPLIT + VAL_SPLIT)) :, :].index, y_test_hat, label="Testing Predictions", color=blmbg_or, linewidth=1.3)
plt.title("LSTM Brent Crude Price Prediction")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.savefig("Algo_Predictions.png", dpi=400)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].plot(y_test_hat, label="Predicted Close Price", color=blmbg_or, linewidth=1.3)
ax[0].plot(y_test, label="True Close Price", color=blmbg_black, linewidth=1.3)
ax[0].set_title("LSTM Brent Crude Price Prediction")
ax[0].set_xlabel("Day")
ax[0].set_ylabel("Price")
ax[0].legend()
ax[1].plot(h.history["loss"], label="Training Loss", color=blmbg_blue, linewidth=1.3)
ax[1].plot(h.history["val_loss"], label="Validation Loss", color=blmbg_red, linewidth=1.3)
ax[1].set_title("LSTM Model Loss")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend()
plt.savefig("Algo_Test_Predictions.png", dpi=400)
plt.show()

# Compute metrics.

mae = mean_absolute_error(y_test, y_test_hat)
mse = mean_squared_error(y_test, y_test_hat)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_hat)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Mean-reversion algorithm

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
actual_price = df["Open BZ=F"].iloc[int(noRows * (TRAIN_SPLIT + VAL_SPLIT)) + 1 :].values
exit_price = df["Open BZ=F"].iloc[int(noRows * (TRAIN_SPLIT + VAL_SPLIT)) + 1 :].shift(-1).values

BASPREAD = df["Open BZ=F"].pct_change().dropna().std()

# S1 = 1 (bid ask spread overrides any potential profit)
S2, S3 = 10, 50

for i in range(int(noRows * (1 - TRAIN_SPLIT - VAL_SPLIT)) - 1):
    pred = predicted_price[i]
    curr = actual_price[i]
    exit = exit_price[i]
    if pred > curr:
        if pred > (1 + BASPREAD)**1.2 * curr:
            buy[i] = S3 * curr + BASPREAD
            buy_exit[i] = S3 * exit 
            buy_plot[i] = curr
            buy_big[i] = curr
        elif pred > (1 + BASPREAD) * curr:
            buy[i] = S2 * curr + BASPREAD
            buy_exit[i] = S2 * exit
            buy_plot[i] = curr
            buy_small[i] = curr
        # else:
        #     buy[i] = S1 * curr
        #     buy_exit[i] = S1 * exit
        #     buy_plot[i] = curr
    else:
        if pred < (1 - BASPREAD)**1.2 * curr:
            sell[i] = S3 * curr
            sell_exit[i] = S3 * exit + BASPREAD
            sell_plot[i] = curr
            sell_big[i] = curr
        elif pred < (1 - BASPREAD) * curr:
            sell[i] = S2 * curr
            sell_exit[i] = S2 * exit + BASPREAD
            sell_plot[i] = curr
            sell_small[i] = curr
        # else:
        #     sell[i] = S1 * curr
        #     sell_exit[i] = S1 * exit
        #     sell_plot[i] = curr

# Compute pnl and plot illustrative graphs

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
# plt.plot(
#     buy_big,
#     color="y",
#     marker="^",
#     linestyle="None",
#     label="3xL Buy $p > (1+\\sigma)^{1.2}a$",
# )
# plt.plot(
#     buy_small,
#     color="g",
#     marker="^",
#     linestyle="None",
#     label="1xL Buy $(1+\\sigma)^{1.2}a > p > (1+\\sigma)a$",
# )
# plt.plot(
#     sell_small,
#     color="r",
#     marker="v",
#     linestyle="None",
#     label="1xL Sell $(1-\\sigma)^{1.2}a < p < (1-\\sigma)a$",
# )
# plt.plot(
#     sell_big,
#     color="m",
#     marker="v",
#     linestyle="None",
#     label="3xL Sell $p < (1-\\sigma)^{1.2}a$",
# )
plt.title("LSTM Mean Reversion Algorithm")
plt.xlabel("Day")
plt.ylabel("Price")
plt.ylim(0.8 * np.min(actual_price), 1.2 * np.max(actual_price))
plt.legend()
plt.savefig("Algo_Signals.png", dpi=400)
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(pnl, color=blmbg_black, linewidth=2)
plt.title("LSTM Mean Reversion Algorithm PnL")
plt.xlabel("Day")
plt.ylabel("PnL")
plt.savefig("Algo_PnL.png", dpi=400)
plt.show()
