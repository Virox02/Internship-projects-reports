import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import mplfinance as mpf
import pmdarima as pm
from scipy.stats import zscore

def SalesPlot():
    # Resample the data by month and sum the 'Sales' for each month

    # Create a line plot of the monthly sales
    df = pd.DataFrame(monthly_sales)
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.title('Monthly Sales Trend')

    # Show the plot
    plt.show()

# print(data.head())
# print(data.isnull().any())

def ProfitPlot():
    # Resample the data by month and sum the 'Profit' for each month
    monthly_profits = data['Profit'].resample('M').sum()

    # Create a line plot of the monthly profits
    df = pd.DataFrame(monthly_profits)
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    plt.xlabel('Month')
    plt.ylabel('Profit')
    plt.title('Monthly Profit Trend')

    # Show the plot
    plt.show()


def SalesCategory():

    # Group the data by 'Category' and resample it by month, summing the 'Sales' for each month
    monthly_sales_by_category = data.groupby('Category')['Sales'].resample('M').sum()
    # Reset the index
    monthly_sales_by_category = monthly_sales_by_category.reset_index()
    # Pivot the dataframe to get 'Category' as columns and 'Sales' as values
    df = monthly_sales_by_category.pivot(index='Order Date', columns='Category', values='Sales')
    # Create a line plot of the monthly sales for each category
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.title('Monthly Sales by Category')
    plt.show()

def Predict(dt):
    # Fit the ARIMA model
    model = pm.auto_arima(dt, seasonal=True, m=12)
    # Make a forecast for the next 12 months
    forecast = model.predict(n_periods=12)
    # Print the forecast
    print(forecast)
    return forecast

def ForecastPlot(forecast, dt, text):
    # Create a range of future dates that is the length of the periods we forecasted
    future_dates = pd.date_range(dt.index[-1], periods = 13, freq='M').tolist()[1:]
    # Convert forecasted data into a pandas series
    forecast_series = pd.Series(forecast, index=future_dates)
    # Plot the sales data
    plt.figure(figsize=(10,6))
    plt.plot(dt)
    plt.plot(forecast_series, linestyle='--')
    plt.xlabel('Time')
    plt.ylabel(text)
    plt.title(f'{text} (Historical and Forecasted)')
    #plt.legend()
    plt.show()

def PredictSalesCategory():
    # Group the data by 'Category' and resample it by month, summing the 'Sales' for each month
    monthly_sales_by_category = data.groupby('Category')['Sales'].resample('M').sum()
    # Get the unique categories
    categories = data['Category'].unique()
    colors = {'Furniture': 'blue', 'Office Supplies': 'orange', 'Technology': 'green'}
    # Create a figure
    plt.figure(figsize=(10,6))
    # Loop over each category
    for i, category in enumerate(categories):
        # Fit the ARIMA model
        model_category = pm.auto_arima(monthly_sales_by_category[category], seasonal=True, m=12)
        # Make a forecast for the next 12 months
        forecast_category = model_category.predict(n_periods=12)
        # Create a range of future dates that is the length of the periods we forecasted
        future_dates_category = pd.date_range(monthly_sales_by_category[category].index[-1], periods = 13, freq='M').tolist()[1:]
        # Convert forecasted data into a pandas series
        forecast_series_category = pd.Series(forecast_category, index=future_dates_category)
        print(forecast_series_category)
        # Plot the sales data for the category
        plt.plot(monthly_sales_by_category[category], label=f'{category} - Historical', color=colors[category])
        plt.plot(forecast_series_category, label=f'{category} - Forecasted', linestyle='--', color=colors[category])

    plt.title('Sales by Category (Historical and Forecasted)')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

def barplt_profit(df):
    profit_by_sub = pd.DataFrame(df.groupby('Sub-Category')['Profit'].sum())
    profit_by_sub = profit_by_sub.reset_index()

    sns.barplot(data=profit_by_sub, x="Profit", y = 'Sub-Category')
    # plt.xticks([0], ['1'], rotation='vertical')
    plt.show()


def pie_profit(df):
    profit_by_cat = pd.DataFrame(df.groupby('Category')['Profit'].sum())
    profit_by_cat = profit_by_cat.reset_index()
    labels = list(profit_by_cat['Category'])
    
    plt.pie(profit_by_cat['Profit'], labels = labels, colors = sns.color_palette('pastel')[0:3], autopct='%.0f%%')
    plt.title("Profit on the basis of Category")
    plt.show()


def segment_sales():
    # Group the data by 'Segment' and resample it by month, summing the 'Sales' for each month
    monthly_sales_by_segment = data.groupby('Segment')['Sales'].resample('M').sum()
    # Reset the index
    monthly_sales_by_segment = monthly_sales_by_segment.reset_index()
    # Pivot the dataframe to get 'Segment' as columns and 'Sales' as values
    df = monthly_sales_by_segment.pivot(index='Order Date', columns='Segment', values='Sales')
    # Create a line plot of the monthly sales for each segment
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.title('Monthly Sales by Segment')
    plt.show()


def segment_profit():
    # Group the data by 'Segment' and resample it by month, summing the 'Profit' for each month
    monthly_profit_by_segment = data.groupby('Segment')['Profit'].resample('M').sum()
    # Reset the index
    monthly_profit_by_segment = monthly_profit_by_segment.reset_index()
    # Pivot the dataframe to get 'Segment' as columns and 'Profit' as values
    df = monthly_profit_by_segment.pivot(index='Order Date', columns='Segment', values='Profit')
    # Create a line plot of the monthly profit for each segment
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    plt.xlabel('Month')
    plt.ylabel('Profit')
    plt.title('Monthly Profit by Segment')
    plt.show()


def boxplot_sales():

    data2 = pd.DataFrame(zscore(data['Sales']))
    # Extract the year from the 'Order Date' index
    data2['Year'] = data2.index.year

    # Create a box plot for Sales by Year
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Year', y='Sales', data=data2)
    plt.title('Yearly Sales Distribution')
    plt.xlabel('Year')
    plt.ylabel('Sales')
    plt.show()





if __name__ == "__main__":

    data = pd.read_excel('Superstore.xls', 'Orders')

    # Convert 'Order Date' to datetime format
    if data['Order Date'].dtype != 'datetime64[ns]':
        data['Order Date'] = pd.to_datetime(data['Order Date'])

    # Set 'Order Date' as the index of the dataframe
    data.set_index('Order Date', inplace=True)

    monthly_sales = data['Sales'].resample('M').sum()
    monthly_profits = data['Profit'].resample('M').sum()

    print(data['Sales'].skew())
    print(data['Profit'].skew())

    # boxplot_sales()
    # boxplot_features()

    # print(data["Sales"].describe())

    # df = pd.DataFrame(monthly_sales.reset_index())

    # sns.boxplot(df, x = 'Order Date', y = 'Sales')
    # plt.show()

    # data2 = pd.DataFrame(zscore(data['Profit']))
    # sns.displot(df, x="Sales", kind="kde")
    # plt.show()

    # boxplot_features()
    # barplt_profit(data)
    # pie_profit(data)
    segment_sales()
    segment_profit()
    # SalesPlot()
    # ProfitPlot()
    # SalesCategory()

    # predicted_sales = Predict(monthly_sales)
    # predicted_profits = Predict(monthly_profits)
    # ForecastPlot(predicted_sales, monthly_sales, "Sales")
    # ForecastPlot(predicted_profits, monthly_profits, "Profits")

    # PredictSalesCategory()
