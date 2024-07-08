import yfinance as yf
import numpy as np
import pandas as pd  # Import pandas for date manipulation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import timedelta

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")  # Fetching 1 year of historical data
    df = df[['Close']]  # We are only interested in the closing prices
    return df

# Function to prepare the data
def prepare_data(df):
    df['Prediction'] = df['Close'].shift(-1)  # Shift the data by one to predict the next day's closing price
    X = np.array(df.drop(['Prediction'], axis=1))[:-1]
    y = np.array(df['Prediction'])[:-1]
    return X, y

# Function to split the data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# Function to train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to make future predictions
def predict_future_prices(model, last_features, days=10):
    future_prices = []
    for _ in range(days):
        future_price = model.predict([last_features])
        future_prices.append(future_price[0])
        last_features = [future_price[0]]
    return future_prices

# GUI Application class
class StockPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Price Predictor")
        self.geometry("800x600")

        self.header = ttk.Label(self, text="STOCK PRICE PREDICTOR", font=("Arial", 16, "bold"))
        self.header.pack(pady=5)

        self.subheader = ttk.Label(self, text="This tool is not financial advice and should not be used to make financial decisions", font=("Arial", 10))
        self.subheader.pack(pady=5)

        self.label = ttk.Label(self, text="Enter Stock Ticker Symbol:")
        self.label.pack(pady=10)

        self.ticker_entry = ttk.Entry(self)
        self.ticker_entry.pack(pady=10)

        self.days_label = ttk.Label(self, text="Enter number of days to predict:")
        self.days_label.pack(pady=10)

        self.days_entry = ttk.Entry(self)
        self.days_entry.pack(pady=10)

        self.predict_button = ttk.Button(self, text="Predict", command=self.predict_stock_price)
        self.predict_button.pack(pady=10)

        self.result_list = tk.Text(self, height=20, width=80, foreground='green', background='black')
        self.result_list.pack(pady=10)

    def predict_stock_price(self):
        ticker = self.ticker_entry.get()
        days = self.days_entry.get()
        if not ticker:
            messagebox.showerror("Error", "Please enter a stock ticker symbol")
            return
        if not days:
            messagebox.showerror("Error", "Please enter the number of days to predict")
            return
        try:
            days = int(days)
            df = fetch_stock_data(ticker)
            X, y = prepare_data(df)
            X_train, X_test, y_train, y_test = split_data(X, y)
            model = train_model(X_train, y_train)
            last_features = [df['Close'].values[-1]]
            future_prices = predict_future_prices(model, last_features, days)

            self.result_list.delete('1.0', tk.END)
            self.result_list.insert(tk.END, f"Predicted stock prices for the next {days} days for {ticker}:\n")
            last_date = df.index[-1]
            for i, price in enumerate(future_prices):
                future_date = last_date + timedelta(days=i+1)
                self.result_list.insert(tk.END, f"{future_date.date()}: ${price:.2f}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = StockPredictorApp()
    app.mainloop()
