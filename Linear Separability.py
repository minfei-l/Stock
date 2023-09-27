import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import confusion_matrix


# take year 1 and examine the plot of your labels. Construct a reduced dataset by removing some green and red points so thatyou can draw a line separating the points. Compute the equation of such a line (many solutiuons are possible)
year1_weeks = pd.read_csv('year1_weeks.csv',encoding='utf-8')[['avg_daily_return','avg_volatility','colour']]

year1_red = year1_weeks[year1_weeks['colour'] == 'red']
year1_green = year1_weeks[year1_weeks['colour'] == 'green']

plt.figure()
plt.scatter(year1_red['avg_daily_return'],year1_red['avg_volatility'],c='r',s=100)
plt.scatter(year1_green['avg_daily_return'],year1_green['avg_volatility'],c='g',s=100)
plt.show()

year1_weeks_removed = year1_weeks.loc[(year1_weeks['avg_daily_return']>-0.02) & (year1_weeks['avg_daily_return']<0.03)].reset_index(drop=True)

year1_red_removed = year1_weeks_removed[year1_weeks_removed['colour'] == 'red']
year1_green_removed = year1_weeks_removed[year1_weeks_removed['colour'] == 'green']

plt.figure()
plt.scatter(year1_red_removed['avg_daily_return'],year1_red_removed['avg_volatility'],c='r',s=100)
plt.scatter(year1_green_removed['avg_daily_return'],year1_green_removed['avg_volatility'],c='g',s=100)
plt.show()

def transform_color(s):
	if s == 'green':
		return 0
	elif s == 'red':
		return 1

year1_weeks_removed['label'] = year1_weeks_removed['colour'].apply(transform_color)

reg = LR().fit(year1_weeks_removed.drop(columns=['colour','avg_volatility']), year1_weeks_removed['avg_volatility'])
reg_intercept = reg.intercept_
reg_coefs = reg.coef_
print(reg_coefs,reg_intercept)

# equation: reg_coefs[0] * x['avg_daily_return'] + reg_intercept

# take this line and use it to assign labels for year 2
year2_weeks = pd.read_csv('year2_weeks.csv',encoding='utf-8')[['avg_daily_return','avg_volatility','colour']]
# year2_weeks['y_pred'] = year2_weeks['avg_daily_return'].apply(lambda x: reg_coefs[0] * x + reg_intercept)

def pred_label(row):
	if row['y_pred'] <= row['avg_volatility']:
		return 'green'
	elif row['y_pred'] > row['avg_volatility']:
		return 'red'

year2_weeks['y_pred'] = year2_weeks['avg_daily_return'].apply(lambda x: reg_coefs[0] * x + reg_intercept)
year2_weeks['label_pred'] = year2_weeks.apply(pred_label,axis=1)
cm = confusion_matrix(year2_weeks['colour'].to_list(),year2_weeks['label_pred'].to_list())#TP FN FP TN
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print('Confusion Matrics: ',cm)
print('Accuracy Score:',accuracy)

def get_night_data(filename):
    stock_data = pd.read_csv(filename + '.csv', encoding='utf-8')

    nights = []
    pre_night_opens = []
    pre_night_closes = []
    post_night_opens = []
    positions = []
    profits = []

    all_money = 100
    values = [100]

    for index, row in stock_data[:-1].iterrows():
        nights.append(row['Date'])
        pre_night_opens.append(row['Open'])
        pre_night_closes.append(row['Close'])
        post_night_opens.append(stock_data['Open'][index + 1])
        if row['Close'] > row['Open']:
            positions.append('long')
            night_profit = round(all_money / row['Close'] * (stock_data['Open'][index + 1] - row['Close']), 2)
            profits.append(night_profit)
            all_money = all_money + night_profit
        elif row['Close'] == row['Open']:
            positions.append('no action')
            profits.append(0)
        else:
            positions.append('short')
            night_profit = round(all_money / row['Close'] * (row['Close'] - stock_data['Open'][index + 1]), 2)
            profits.append(night_profit)
            all_money = all_money + night_profit

        values.append(all_money)

    night_data = pd.DataFrame(
        columns=['night', 'pre_night_open', 'pre_night_close', 'post_night_open', 'position', 'profit'])
    night_data['night'] = nights
    night_data['pre_night_open'] = pre_night_opens
    night_data['pre_night_close'] = pre_night_closes
    night_data['post_night_open'] = post_night_opens
    night_data['position'] = positions
    night_data['profit'] = profits
    # night_data['value'] = values

    # print(night_data)
    night_data.to_csv(filename + "_night.csv", index=False)

    print('Answer1: The average nightly profit for "{}" is: {}'.format(filename, round(night_data['profit'].mean(), 2)))

    print(
        'Answer2: The profit from "long" positions for {} is: {} \n\t The profit from "short" positions for {} is: {} '.format(
            filename, round(sum(night_data[night_data['position'] == 'long']['profit']), 2), filename,
            round(sum(night_data[night_data['position'] == 'short']['profit']), 2)))

    bh_share = round(100 / stock_data['Open'][0], 2)
    bh_value_lst = []
    for index, row in stock_data.iterrows():
        bh_value_lst.append(bh_share * row['Close'])

    return values, bh_value_lst

