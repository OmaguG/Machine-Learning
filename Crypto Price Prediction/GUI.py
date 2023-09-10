import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten
from sklearn.metrics import mean_squared_error
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

def main():
    st.title("SOLiGence Investment LTD")
    # Company logo
    logo_image = "logo.png"  # Replace with the path to your logo image file
    st.sidebar.image(logo_image, use_column_width=True)

    symbols = ['XRP-USD', 'ADA-USD', 'MATIC-USD', 'DOT-USD', 'TON-USD', 'DASH-USD', 'FRAX-USD', 'LINK-USD', 'WBTC-USD',
               'MANA-USD', 'CAKE-USD', 'BEAM-USD', 'ETC-USD', 'TWT-USD', 'BTG-USD', 'DOGE-USD', 'DAI-USD', 'LEO-USD', 'AVAX-USD']

    df = download_and_concatenate_data(symbols)

    if st.checkbox("Show Data"):
        st.write(df)

    if st.button("Train Model"):
        model = train_model(df)
        st.session_state['model'] = model
        st.success("Model trained successfully!")

    timeframe_options = ["Today", "Next Week", "Next Month", "Quarter"]
    selected_timeframe = st.selectbox("Select Timeframe", timeframe_options)

    if st.button("Make Predictions"):
        if 'model' not in st.session_state:
            st.error("Please train the model first!")
        else:
            model = st.session_state['model']
            investment_amount = st.number_input("Enter Investment Amount (in dollars)")
            expected_profit = st.number_input("Enter Expected Profit (in dollars)")
            combinations = find_best_combinations(model, df, investment_amount, expected_profit, selected_timeframe)
            if combinations:
                st.success("Best combinations found!")
                st.write(combinations)
            else:
                st.error("No combinations found to meet the criteria.")

    if st.button("Calculate Errors"):
        if 'model' not in st.session_state:
            st.error("Please train the model first!")
        else:
            model = st.session_state['model']
            errors = calculate_errors(model, df)
            st.success("Errors calculated successfully!")
            st.write(errors)

def download_and_concatenate_data(symbols):
    df = pd.DataFrame()

    for symbol in symbols:
        data = yf.download(symbol, group_by='symbols', period='max')
        data['Name'] = symbol
        df = pd.concat([df, data])

    return df

def train_model(df):
    X_train, X_test = df['Close'].values[:365], df['Close'].values[365:]
    X_train = X_train.reshape(365, 1, 1)
    X_test = X_test.reshape(-1, 1, 1)

    model = Sequential([
        Conv1D(64, 3, padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Flatten(),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, X_train, epochs=18)

    return model

def make_predictions(model, df, timeframe):
    if timeframe == "Today":
        prediction_date = pd.Timestamp.now()
    elif timeframe == "Next Week":
        prediction_date = pd.Timestamp.now() + pd.DateOffset(weeks=1)
    elif timeframe == "Next Month":
        prediction_date = pd.Timestamp.now() + pd.DateOffset(months=1)
    elif timeframe == "Quarter":
        prediction_date = pd.Timestamp.now() + pd.DateOffset(months=3)

    df['Predicted'] = np.nan
    df['Prediction Date'] = prediction_date

    return df[['Name', 'Close', 'Predicted', 'Prediction Date']]

def calculate_errors(model, df):
    X_test = df['Close'].values[365:]
    X_test = X_test.reshape(-1, 1, 1)

    predictions = model.predict(X_test)

    rmse_scores = [np.sqrt(mean_squared_error(X_test[i], predictions[i])) for i in range(len(X_test))]
    mse_scores = [mean_squared_error(X_test[i], predictions[i]) for i in range(len(X_test))]

    errors = pd.DataFrame({'Symbol': df['Name'][365:], 'RMSE': rmse_scores, 'MSE': mse_scores})

    return errors

def calculate_profit(selected_df, investment_amount, timeframe):
    total_investment = investment_amount / len(selected_df)
    predicted_prices = selected_df['Predicted'].values
    current_prices = selected_df['Close'].values

    if timeframe == "Today":
        profit = total_investment * (predicted_prices[-1] - current_prices[-1])
    else:
        profit = total_investment * (predicted_prices[-1] - current_prices[0])

    return profit

def find_best_combinations(model, df, investment_amount, expected_profit, timeframe):
    coins = df['Name'].unique()
    max_coin_combinations = 5  # Set the maximum number of coins in a combination
    combinations_found = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        
        for r in range(1, min(max_coin_combinations, len(coins)) + 1):
            for subset in combinations(coins, r):
                selected_df = df[df['Name'].isin(subset)].copy()
                futures.append(executor.submit(make_predictions, model, selected_df, timeframe))
        
        for future in futures:
            predictions_df = future.result()
            if predictions_df is not None:
                profit = calculate_profit(predictions_df, investment_amount, timeframe)
                if profit >= expected_profit:
                    combinations_found.append({
                        'Coins': predictions_df['Name'].unique(),
                        'Investment per Coin': investment_amount / len(predictions_df['Name'].unique()),
                        'Profit': profit
                    })
    
    return combinations_found

if __name__ == "__main__":
    main()
