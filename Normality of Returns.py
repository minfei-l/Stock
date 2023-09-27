import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_return_type(rt_num):
    if rt_num >= 0:
        return 'pos'
    else:
        return 'neg'


data = pd.read_csv('LNVGF.csv', encoding='utf-8')
data['return_type'] = data['Return'].apply(add_return_type)

# for every year (out of 5 years), compute the number of days with positive and negative returns.
# for each year, compute the average of daily returns μ and compute the percentage of days with returns greater than μ and the proportion of days with returns less than μ.
years = []
trading_days_nums = []
positive_days_nums = []
negative_days_nums = []
avg_daily_returns = []
std_daily_returns = []

greater_days_pers = []
less_days_pers = []

greater_pos_days_pers = []
greater_neg_days_pers = []
less_pos_days_pers = []
less_neg_days_pers = []

greater_2sig_days_pers = []
less_2sig_days_pers = []

greater_2sig_pos_days_pers = []
greater_2sig_neg_days_pers = []
less_2sig_pos_days_pers = []
less_2sig_neg_days_pers = []

for year_data in data.groupby('Year'):
    years.append(year_data[0])

    trading_days_nums.append(year_data[1].shape[0])

    positive_days_num = year_data[1][year_data[1]['return_type'] == 'pos'].shape[0]
    positive_days_nums.append(positive_days_num)

    negative_days_num = year_data[1][year_data[1]['return_type'] == 'neg'].shape[0]
    negative_days_nums.append(negative_days_num)

    avg = year_data[1]['Return'].mean()
    std = year_data[1]['Return'].std()
    avg_daily_returns.append(avg)
    std_daily_returns.append(std)

    greater_days_pers.append(year_data[1][year_data[1]['Return'] > avg].shape[0] / year_data[1].shape[0])
    less_days_pers.append(year_data[1][year_data[1]['Return'] < avg].shape[0] / year_data[1].shape[0])

    greater_pos_days_pers.append(
        year_data[1][(year_data[1]['Return'] > avg) & (year_data[1]['return_type'] == 'pos')].shape[0] /
        year_data[1].shape[0])
    greater_neg_days_pers.append(
        year_data[1][(year_data[1]['Return'] > avg) & (year_data[1]['return_type'] == 'neg')].shape[0] /
        year_data[1].shape[0])
    less_pos_days_pers.append(
        year_data[1][(year_data[1]['Return'] < avg) & (year_data[1]['return_type'] == 'pos')].shape[0] /
        year_data[1].shape[0])
    less_neg_days_pers.append(
        year_data[1][(year_data[1]['Return'] < avg) & (year_data[1]['return_type'] == 'neg')].shape[0] /
        year_data[1].shape[0])

    greater_2sig_days_pers.append(year_data[1][year_data[1]['Return'] > avg + 2 * std].shape[0] / year_data[1].shape[0])
    less_2sig_days_pers.append(year_data[1][year_data[1]['Return'] < avg - 2 * std].shape[0] / year_data[1].shape[0])

    greater_2sig_pos_days_pers.append(
        year_data[1][(year_data[1]['Return'] > avg + 2 * std) & (year_data[1]['return_type'] == 'pos')].shape[0] /
        year_data[1].shape[0])
    greater_2sig_neg_days_pers.append(
        year_data[1][(year_data[1]['Return'] > avg + 2 * std) & (year_data[1]['return_type'] == 'neg')].shape[0] /
        year_data[1].shape[0])
    less_2sig_pos_days_pers.append(
        year_data[1][(year_data[1]['Return'] < avg - 2 * std) & (year_data[1]['return_type'] == 'pos')].shape[0] /
        year_data[1].shape[0])
    less_2sig_neg_days_pers.append(
        year_data[1][(year_data[1]['Return'] < avg - 2 * std) & (year_data[1]['return_type'] == 'neg')].shape[0] /
        year_data[1].shape[0])

for i in range(len(years)):
    print('Year:', years[i])
    print('The number of trading days:', trading_days_nums[i])
    print('The number of days with positive returns:', positive_days_nums[i])
    print('The number of days with negative returns:', negative_days_nums[i])
    print('The average of daily returns:', avg_daily_returns[i])
    print('The standard deviation of daily returns:', std_daily_returns[i])
    print('The percentage of days with returns greater than μ:', greater_days_pers[i])
    print('The percentage of days with returns less than μ:', less_days_pers[i])
    print('The percentage of days with positive returns greater than μ:', greater_pos_days_pers[i])
    print('The percentage of days with positive returns less than μ:', less_pos_days_pers[i])
    print('The percentage of days with negative returns greater than μ:', greater_neg_days_pers[i])
    print('The percentage of days with negative returns less than μ:', less_neg_days_pers[i])
    print('The percentage of days with returns more than 2 std from the mean:', greater_2sig_days_pers[i])
    print('The percentage of days with returns less than 2 std from the mean:', less_2sig_days_pers[i])
    print('The percentage of days with positive returns more than 2 std from the mean:', greater_2sig_pos_days_pers[i])
    print('The percentage of days with negative returns more than 2 std from the mean:', greater_2sig_neg_days_pers[i])
    print('The percentage of days with positive returns less than 2 std from the mean:', less_2sig_pos_days_pers[i])
    print('The percentage of days with negative returns less than 2 std from the mean:', less_2sig_neg_days_pers[i])
    print('------------------------------------------------------------------------------------------------------\n')


