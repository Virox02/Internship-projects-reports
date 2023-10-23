import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import mplfinance as mpf
import pmdarima as pm
from scipy.stats import zscore


def distribution_features(df):

    # Box plot for Sales
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df['Sales'])
    plt.title('Box Plot distribution for Sales')
    plt.show()

    # Box plot for Profit
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df['Profit'])
    plt.title('Box Plot distribution for Profit')
    plt.show()

    # KDE plot for Sales
    sns.displot(df, x="Sales", kind="kde")
    plt.title('KDE for Sales')
    plt.show()

    # KDE plot for Profit
    sns.displot(df, x="Profit", kind="kde")
    plt.title('KDE for Profit')
    plt.show()



def remove_outliers():
    # SALES OUTLIERS
    # Calculate Q1 and Q3 for Sales
    Q1_sales = data['Sales'].quantile(0.25)
    Q3_sales = data['Sales'].quantile(0.75)
    IQR_sales = Q3_sales - Q1_sales
    lower_bound_sales = Q1_sales - 1.5 * IQR_sales
    upper_bound_sales = Q3_sales + 1.5 * IQR_sales

    # PROFIT OUTLIERS
    # Calculate Q1 and Q3 for Profit
    Q1_profit = data['Profit'].quantile(0.25)
    Q3_profit = data['Profit'].quantile(0.75)
    IQR_profit = Q3_profit - Q1_profit
    lower_bound_profit = Q1_profit - 1.5 * IQR_profit
    upper_bound_profit = Q3_profit + 1.5 * IQR_profit

    # Remove outliers for both Sales and Profit
    filtered_data = data[(data['Sales'] >= lower_bound_sales) & (data['Sales'] <= upper_bound_sales) &
                        (data['Profit'] >= lower_bound_profit) & (data['Profit'] <= upper_bound_profit)]

    return filtered_data


def barplt_profit(df):
    profit_by_sub = pd.DataFrame(df.groupby('Sub-Category')['Profit'].sum())
    profit_by_sub = profit_by_sub.reset_index()

    ax = sns.barplot(data=profit_by_sub, x="Profit", y = 'Sub-Category')
    ax.bar_label(ax.containers[0])
    # plt.xticks([0], ['1'], rotation='vertical')
    plt.title("Profit on the basis of Sub-Category")
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
    monthly_sales_by_segment = new_data.groupby('Segment')['Sales'].resample('M').sum()
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
    monthly_profit_by_segment = new_data.groupby('Segment')['Profit'].resample('M').sum()
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


def boxplot_sales(df):

    # Extract the year from the 'Order Date' index
    df['Year'] = df.index.year

    # Create a box plot for Sales by Year
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Year', y='Sales', data=df)
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

    # monthly_sales = data['Sales'].resample('M').sum()
    # monthly_profits = data['Profit'].resample('M').sum()

    # print(data['Sales'].skew())
    # print(data['Profit'].skew())

    new_data = pd.DataFrame(remove_outliers())

    distribution_features(data)
    distribution_features(new_data)

    print(data["Sales"].describe())
    print(new_data["Sales"].describe())
    print(data["Profit"].describe())
    print(new_data["Profit"].describe())

    barplt_profit(new_data)
    pie_profit(new_data)
    segment_sales()
    segment_profit()
    boxplot_sales(new_data)

