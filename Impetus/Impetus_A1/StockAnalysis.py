import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# loading the dataset
data = pd.read_csv("stocks.csv")
#print(data.describe())

data['Date'] = pd.to_datetime(data['Date'])

''' 
Plot for Adjacent Closing Prices Over The Past 3 Months For Each Company.
This plot visualizes the price trends.
'''
def priceTrendsPlot():
    plt.figure(figsize=(14, 8))
    for ticker in data['Ticker'].unique():
        plt.plot(data[data['Ticker'] == ticker]['Date'], 
                data[data['Ticker'] == ticker]['Adj Close'],
                label=ticker)
    plt.xlabel('Date')
    plt.ylabel('Adj Close Price')
    plt.title('Adjacent Closing Prices Over Time')
    plt.legend()
    plt.show()


'''
Plot that shows the volume of stocks traded by these companies, in millions, over time.
'''
def VolumeTrendsPlot():
    plt.figure(figsize=(14, 8))
    for ticker in data['Ticker'].unique():
        plt.plot(data[data['Ticker'] == ticker]['Date'], 
                data[data['Ticker'] == ticker]['Volume'] / 1e6,
                label=ticker)
    plt.xlabel('Date')
    plt.ylabel('Volume (in millions)')
    plt.title('Volume of Stocks Traded Over Time')
    plt.legend()
    plt.show()

'''
Function to calculate a 7-day moving averages for each company, and visualize it
over time.
'''
def MovingAvgPlot():
    for ticker in data['Ticker'].unique():
        data.loc[data['Ticker'] == ticker, '7-day MA'] = data[data['Ticker'] == ticker]['Adj Close'].rolling(window=7).mean()

    plt.figure(figsize=(14, 8))
    for ticker in data['Ticker'].unique():
        plt.plot(data[data['Ticker'] == ticker]['Date'], 
                data[data['Ticker'] == ticker]['7-day MA'], 
                label=ticker)
    plt.xlabel('Date')
    plt.ylabel('7-day Moving Average')
    plt.title('7-day Moving Averages Over Time')
    plt.legend()
    plt.show()

'''
Function to calculate the volatility for each company, and visualize it over time.
 '''   
def volatilityPlot():
    for ticker in data['Ticker'].unique():
        data.loc[data['Ticker'] == ticker, 'Volatility'] = data[data['Ticker'] == ticker]['Adj Close'].pct_change().rolling(window=7).std()

    plt.figure(figsize=(14, 8))
    for ticker in data['Ticker'].unique():
        plt.plot(data[data['Ticker'] == ticker]['Date'], 
                data[data['Ticker'] == ticker]['Volatility'], 
                label=ticker)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title('Volatility Over Time')
    plt.legend()
    plt.show()

'''
Correlation Matrix plot to examine the relationships between different stock prices.
'''
def corrPlot():
    corr_matrix = data.pivot_table(index='Date', columns='Ticker', values='Adj Close').corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

'''
Pair plot of the daily returns of each stock. This can help us see how the daily 
returns of each pair of stocks relate to each other.
'''
def dailyReturnPairPlot():
    for ticker in data['Ticker'].unique():
        data.loc[data['Ticker'] == ticker, 'Daily Return'] = data[data['Ticker'] == ticker]['Adj Close'].pct_change()

    daily_returns = data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
    sns.pairplot(daily_returns.dropna())
    plt.title('Daily Returns')
    plt.show()

if __name__ == "__main__":

    priceTrendsPlot()
    VolumeTrendsPlot()
    MovingAvgPlot()
    volatilityPlot()
    corrPlot()
    dailyReturnPairPlot()