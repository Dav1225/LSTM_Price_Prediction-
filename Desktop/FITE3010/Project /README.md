# LSTM Model for Predicting SPY Next-Day Log Returns

This project implements an LSTM (Long Short-Term Memory) model to predict the next-day log returns of SPY (S&P 500 ETF). The notebook preprocesses data, trains the model, evaluates its performance, and visualizes the results.

## Project Structure
- **`LSTM_SPY.ipynb`**: Jupyter Notebook containing the implementation of the LSTM model.
- **`LSTM_SPY_21_Nov.csv`**: Dataset used for training and testing the model.
- **`best_model.keras`**: Saved model file containing the best weights during training.

## Features
The dataset includes the following features:
- **Price Data**: Close, High, Low, Open.
- **Trend Indicators**: DIF, MACD, Efficiency Ratio (20D).
- **Momentum and Oscillators**: RSI (6, 12, 24), Overbought, Oversold.
- **Volatility and Market Structure**: Volume, Volatility (20D), Position in Bollinger Bands, Log High-Low Ratio.
- **Recent Performance**: Log Return, Log Return (Intraday), Log Return (5D).

## Workflow
### 1. Data Preprocessing
- Load the dataset and ensure the `Date` column is in datetime format.
- Split the data into training and testing sets based on a specific date (`2022-01-03`).
- Scale the features using `MinMaxScaler`.
- Create sequences for the LSTM model with a sequence length of 9.

### 2. Model Architecture
The LSTM model consists of:
- 3 LSTM layers with 256, 128, and 32 units respectively.
- Dropout layers to prevent overfitting.
- A Dense layer for the final output.
- Custom loss function: Mean Squared Error (MSE) with a boldness adjustment penalty.

### 3. Training
- The model is trained for up to 50 epochs with a batch size of 4.
- Early stopping, learning rate reduction, and model checkpointing are used as callbacks.

### 4. Evaluation
- Evaluate the model on the test set using MSE.
- Visualize actual vs predicted log returns for both training and testing sets.
- Convert predicted log returns to next-day close prices and calculate the MSE.

### 5. Optimization
- Find the optimal multiplier `n` to adjust predicted log returns and minimize MSE.
- Visualize actual vs adjusted predicted next-day close prices.

## Results
- The model achieves a low MSE on the test set.
- The optimal multiplier `n` further improves the prediction accuracy.
- Visualizations show the model's performance in predicting SPY next-day log returns and close prices.

## How to Run
1. Install the required Python libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow
   ```
2. Open the `LSTM_SPY.ipynb` notebook in Jupyter Notebook or VS Code.
3. Run all cells to preprocess data, train the model, and evaluate its performance.

## Dependencies
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, scikit-learn, tensorflow

## Acknowledgments
This project uses historical SPY data and various technical indicators to predict next-day log returns. The LSTM model is trained and evaluated using TensorFlow.

## License
This project is licensed under the MIT License. See the LICENSE file for details.