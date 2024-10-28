import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from keras import layers
from art import text2art

# Function to print the introduction of the program
def print_intro():
    ascii_art = text2art("LotteryAi")
    print("============================================================")
    print("LotteryAi")
    print("Created by: Corvus Codex")
    print("Github: https://github.com/CorvusCodex/")
    print("Licence : MIT License")
    print("Support my work:")
    print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
    print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
    print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
    print("============================================================")
    print(ascii_art)
    print("Lottery prediction artificial intelligence")

# Load data for Random Forest prediction
def load_rf_data(filename='se.csv'):
    data = pd.read_csv(filename)
    columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'NumS']
    return data[columns]

# Load data for LSTM model
def load_lstm_data(filename='data.txt'):
    data = np.genfromtxt(filename, delimiter=',', dtype=int)
    data[data == -1] = 0
    train_data = data[:int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)):]
    max_value = np.max(data)
    return train_data, val_data, max_value

# Random Forest prediction function
def random_forest_prediction(data):
    columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'NumS']
    X = data[columns].iloc[:-1]
    y = data[columns].iloc[1:]
    classifiers = {}
    predictions = []
    selected_numbers = set()
    
    for col in columns[:-1]:
        clf = RandomForestClassifier(max_depth=2, n_estimators=300)
        clf.fit(X[[col]], y[col])
        prob = clf.predict_proba(X[[col]].iloc[-1:])[0]
        sorted_indices = np.argsort(-prob)
        
        chosen = False
        for idx in sorted_indices:
            num = idx + 1
            if prob[idx] >= 0.6 and num not in selected_numbers:
                predictions.append(num)
                selected_numbers.add(num)
                chosen = True
                break
        if not chosen:
            for idx in sorted_indices:
                num = idx + 1
                if num not in selected_numbers:
                    predictions.append(num)
                    selected_numbers.add(num)
                    break
    
    clf_s = RandomForestClassifier(max_depth=2, n_estimators=300)
    clf_s.fit(X[['NumS']], y['NumS'])
    prob_s = clf_s.predict_proba(X[['NumS']].iloc[-1:])[0]
    sorted_indices_s = np.argsort(-prob_s)
    next_draw_s_prob = None
    for idx in sorted_indices_s:
        special_num = idx + 1
        if prob_s[idx] >= 0.6 and special_num not in selected_numbers:
            next_draw_s_prob = special_num
            break
    if next_draw_s_prob is None:
        for idx in sorted_indices_s:
            special_num = idx + 1
            if special_num not in selected_numbers:
                next_draw_s_prob = special_num
                break

    return predictions, next_draw_s_prob

# LSTM prediction function
def lstm_prediction(train_data, val_data, max_value):
    num_features = train_data.shape[1]
    model = keras.Sequential([
        layers.Embedding(input_dim=max_value + 1, output_dim=64),
        layers.LSTM(256),
        layers.Dense(num_features, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=200, verbose=0)
    predictions = model.predict(val_data)
    indices = np.argsort(predictions, axis=1)[:, -num_features:]
    predicted_numbers = np.take_along_axis(val_data, indices, axis=1)
    return predicted_numbers[0]

# Main function to execute both predictions and print results
def main():
    print_intro()
    
    # Load data for both models
    rf_data = load_rf_data()
    lstm_train_data, lstm_val_data, lstm_max_value = load_lstm_data()
    
    # Random Forest Prediction
    rf_predictions, rf_special = random_forest_prediction(rf_data)
    # print("Random Forest Predicted Numbers:", rf_predictions, "Special number:", rf_special)
    st.write("Random Forest Predicted Numbers:", rf_predictions, "Special number:", rf_special)
    
    # LSTM Prediction
    lstm_predictions = lstm_prediction(lstm_train_data, lstm_val_data, lstm_max_value)
    #print("LSTM Predicted Numbers:", ', '.join(map(str, lstm_predictions)))
    st.write("LSTM Predicted Numbers:", ', '.join(map(str, lstm_predictions)))

# Run main function if this script is run directly
if __name__ == "__main__":
    main()
