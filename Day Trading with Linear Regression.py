import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def output_result(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  # TP FN FP TN
    tpr = cm[0][0] / (cm[0][0] + cm[0][1])
    tnr = cm[1][1] / (cm[1][1] + cm[1][0])
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print('Confusion Matrics: ', cm)
    print('Accuracy Score:', accuracy)
    print('TPR: ', tpr)
    print('TNR: ', tnr)
    print('----------------------------------------------------------------')
    return accuracy


# take W = 5,6,...,30 and consider your data for year 1. For each W in the specified range, compute your average P/L per trade and plot it: on x-axis you plot the values of W and on the y axis you plot profit and loss per trade. What is the optimal value W⇤ of W?
year1_data = pd.read_csv('year1.csv')[['Date', 'Adj Close', 'Open', 'Return']]

avg_profits = []

for w in range(5, 31):
    pred_prices = []
    transactions_num = 0

    position = '0'
    all_money = 100
    share = 0.0

    positions = []
    all_moneys = []
    shares = []

    for index, row in year1_data.iterrows():
        if index <= w - 1:  # 前w天的closing prices不做预测
            pred_prices.append(row['Adj Close'])
        else:
            X = np.array(range(index - w + 1, index + 1)).reshape(w, 1)
            Y = year1_data.drop(['Date', 'Open', 'Return'], axis=1).iloc[index - w:index].reset_index(drop=True)
            model = LinearRegression().fit(X, Y)
            pred_value = model.predict([[index]])
            pred_prices.append(pred_value[0][0])

            if pred_value[0][0] > year1_data['Adj Close'][index - 1]:
                if position == '0':
                    share = all_money / year1_data['Adj Close'][index - 1]
                    all_money = 0
                    position = 'long'
                elif position == 'long':
                    pass
                elif position == 'short':
                    all_money = all_money - share * year1_data['Adj Close'][index - 1]
                    share = 0
                    position = '0'
                    transactions_num = transactions_num + 1
            if pred_value[0][0] < year1_data['Adj Close'][index - 1]:
                if position == '0':
                    share = all_money / year1_data['Adj Close'][index - 1]
                    all_money = all_money * 2
                    position = 'short'
                elif position == 'short':
                    pass
                elif position == 'long':
                    all_money = all_money + share * year1_data['Adj Close'][index - 1]
                    share = 0
                    position = '0'
                    transactions_num = transactions_num + 1
            if pred_value[0][0] == year1_data['Adj Close'][index - 1]:
                pass

        positions.append(position)
        all_moneys.append(all_money)
        shares.append(share)

    if position == '0':
        all_money = all_money
    elif position == 'long':
        all_money = all_money + share * list(year1_data['Adj Close'])[-1]
    elif position == 'short':
        all_money = all_money - share * list(year1_data['Adj Close'])[-1]

    avg_profit = (all_money - 100) / transactions_num
    avg_profits.append(avg_profit)
    print('When W={}, the average P/L per trade is {}'.format(w, avg_profit))

plt.plot(range(5, 31), avg_profits)
plt.show()

# use the value of optimal W from year 1 and consider year 2.
# W=16
year2_data = pd.read_csv('year2.csv')[['Date', 'Adj Close', 'Open', 'Return']]

w = 16

pred_prices = []

position = '0'
all_money = 100
share = 0.0
transaction = '0'
profit = 0
r2 = 0

positions = []
all_moneys = []
shares = []
transactions = []
profits = []
r2s = []

for index, row in year2_data.iterrows():
    if index <= w - 1:  # 前w天的closing prices不做预测
        pred_prices.append(row['Adj Close'])
    else:
        X = np.array(range(index - w + 1, index + 1)).reshape(w, 1)
        Y = year2_data.drop(['Date', 'Open', 'Return'], axis=1).iloc[index - w:index].reset_index(drop=True)
        model = LinearRegression().fit(X, Y)
        r2 = model.coef_[0][0] ** 2
        pred_value = model.predict([[index]])
        pred_prices.append(pred_value[0][0])

        if pred_value[0][0] > year2_data['Adj Close'][index - 1]:
            if position == '0':
                share = all_money / year2_data['Adj Close'][index - 1]
                all_money = 0
                position = 'long'
                transaction = '0'
                profit = 0
            elif position == 'long':
                transaction = '0'
                profit = 0
                pass
            elif position == 'short':
                now_money = all_money - share * year2_data['Adj Close'][index - 1]
                profit = now_money - all_money / 2
                all_money = now_money
                share = 0
                position = '0'
                transaction = 'short position trade'

        if pred_value[0][0] < year2_data['Adj Close'][index - 1]:
            if position == '0':
                share = all_money / year2_data['Adj Close'][index - 1]
                all_money = all_money * 2
                position = 'short'
                transaction = '0'
                profit = 0
            elif position == 'short':
                transaction = '0'
                profit = 0
                pass
            elif position == 'long':
                now_money = all_money + share * year2_data['Adj Close'][index - 1]
                last_buy_index = len(positions) - 1 - positions[::-1].index('0')
                profit = now_money - all_moneys[last_buy_index]
                all_money = now_money
                share = 0
                position = '0'
                transaction = 'long position trade'
        if pred_value[0][0] == year2_data['Adj Close'][index - 1]:
            transaction = '0'
            profit = 0
            pass

    # print(row['Adj Close'],pred_prices[index],position,transaction,all_money,share,profit)
    positions.append(position)
    all_moneys.append(all_money)
    shares.append(share)
    transactions.append(transaction)
    profits.append(profit)
    r2s.append(r2)

print('The average r^2 is:', pd.Series(r2s).mean())

plt.plot(range(1, year2_data.shape[0] + 1), r2s)
plt.show()

# what is the average profit/loss per ”long position” trade and per ”short position” trades in year 2?
long_trade_profits = []
short_trade_profits = []
long_trans_days = []
short_trans_days = []

for i in range(len(transactions)):
    if transactions[i] == 'long position trade':
        long_trade_profits.append(profits[i])
        last_0_index = len(positions[:i]) - 1 - positions[:i][::-1].index('0')
        long_trans_days.append(i - last_0_index)
    if transactions[i] == 'short position trade':
        short_trade_profits.append(profits[i])
        last_0_index = len(positions[:i]) - 1 - positions[:i][::-1].index('0')
        short_trans_days.append(i - last_0_index)

print('There are {} "long position" transactions in year2'.format(len(long_trade_profits)))
print('There are {} "short position" transactions in year2'.format(len(short_trade_profits)))

print('The average profit/loss per "long position" trade in year2 is:',
      sum(long_trade_profits) / len(long_trade_profits))
print('The average profit/loss per "short position" trade in year2 is:',
      sum(short_trade_profits) / len(short_trade_profits))

# what is the average number of days for long position and short position transactions in year 2?
print('The average number of days for long position transactions in year2 is:',
      sum(long_trans_days) / len(long_trans_days))
print('The average number of days for short position transactions in year2 is:',
      sum(short_trans_days) / len(short_trans_days))

if position == '0':
    all_money = all_money
elif position == 'long':
    all_money = all_money + share * list(year2_data['Adj Close'])[-1]
elif position == 'short':
    all_money = all_money - share * list(year2_data['Adj Close'])[-1]

avg_profit = (all_money - 100) / (len(long_trade_profits) + len(short_trade_profits))
print('When W={}, the average P/L per trade in year2 is {}'.format(w, avg_profit))



