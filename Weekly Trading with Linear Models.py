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


# take weekly data for year 1. For each W = 5,6,...,12 and...
year1_data = pd.read_csv('year1_weeks.csv')[['week_close', 'colour']]

accs = []
for d in [1, 2, 3]:
    d_accs = []
    for w in [5, 6, 7, 8, 9, 10, 11, 12]:
        pred_list = []
        labels = []
        for index, row in year1_data.iterrows():
            if index <= w - 1:  # the closing price in the previous w weeks, we will not predict them
                pred_list.append(row['week_close'])
                labels.append(row['colour'])
            else:
                X = np.array(range(index + 1 - w, index + 1)).reshape(w, 1)
                # print(X)
                Y = year1_data.drop(['colour'], axis=1).iloc[index - w:index].reset_index(drop=True)
                # print(Y)
                poly_reg = PolynomialFeatures(degree=d)
                X_poly = poly_reg.fit_transform(X)
                # print(poly_reg.get_feature_names())
                model = LinearRegression().fit(X_poly, Y)
                pred_value = model.predict(poly_reg.fit_transform([[index]]))
                # print(row['week_close'],pred_value)
                pred_list.append(pred_value[0][0])

                if pred_value[0][0] > year1_data['week_close'][index - 1]:
                    labels.append('green')
                if pred_value[0][0] < year1_data['week_close'][index - 1]:
                    labels.append('red')
                if pred_value[0][0] == year1_data['week_close'][index - 1]:
                    labels.append(labels[index - 1])

        # break

        print('Degree:{}\tW:{}'.format(d, w))
        acc = output_result(list(year1_data['colour'])[w:], labels[w:])
        d_accs.append(acc)
    accs.append(d_accs)

plt.plot([5, 6, 7, 8, 9, 10, 11, 12], accs[0], label='d=1')
plt.plot([5, 6, 7, 8, 9, 10, 11, 12], accs[1], label='d=2')
plt.plot([5, 6, 7, 8, 9, 10, 11, 12], accs[2], label='d=3')
plt.legend()
plt.show()

# for each d take the best W that gives you the highest accu- racy. Use this W to predict labels for year 2. What is your accuracy?
year2_data = pd.read_csv('year2_weeks.csv')[['week_open', 'week_close', 'colour']]

# d=1,w=5
pred_list = []
labels1 = []
for index, row in year2_data.iterrows():
    if index <= 4:  # the closing price in the previous 5 weeks, we will not predict them
        pred_list.append(row['week_close'])
        labels1.append(row['colour'])
    else:
        X = np.array(range(index - 4, index + 1)).reshape(5, 1)
        Y = year2_data.drop(['week_open', 'colour'], axis=1).iloc[index - 5:index].reset_index(drop=True)
        poly_reg = PolynomialFeatures(degree=1)
        X_poly = poly_reg.fit_transform(X)
        model = LinearRegression().fit(X_poly, Y)
        pred_value = model.predict(poly_reg.fit_transform([[index]]))
        pred_list.append(pred_value[0][0])
        if pred_value[0][0] > year2_data['week_close'][index - 1]:
            labels1.append('green')
        if pred_value[0][0] < year2_data['week_close'][index - 1]:
            labels1.append('red')
        if pred_value[0][0] == year2_data['week_close'][index - 1]:
            labels1.append(labels1[index - 1])

print('Degree:{}\tW:{}'.format(1, 5))
acc = output_result(list(year2_data['colour'])[5:], labels1[5:])

# d=2,w=8
pred_list = []
labels2 = []
for index, row in year2_data.iterrows():
    if index <= 7:  # the closing price in the previous 8 weeks, we will not predict them
        pred_list.append(row['week_close'])
        labels2.append(row['colour'])
    else:
        X = np.array(range(index - 7, index + 1)).reshape(8, 1)
        Y = year2_data.drop(['week_open', 'colour'], axis=1).iloc[index - 8:index].reset_index(drop=True)
        poly_reg = PolynomialFeatures(degree=2)
        X_poly = poly_reg.fit_transform(X)
        model = LinearRegression().fit(X_poly, Y)
        pred_value = model.predict(poly_reg.fit_transform([[index]]))
        pred_list.append(pred_value[0][0])
        if pred_value[0][0] > year2_data['week_close'][index - 1]:
            labels2.append('green')
        if pred_value[0][0] < year2_data['week_close'][index - 1]:
            labels2.append('red')
        if pred_value[0][0] == year2_data['week_close'][index - 1]:
            labels2.append(labels2[index - 1])

print('Degree:{}\tW:{}'.format(2, 8))
acc = output_result(list(year2_data['colour'])[8:], labels2[8:])

# d=3,w=10
pred_list = []
labels3 = []
for index, row in year2_data.iterrows():
    if index <= 9:  # the closing price in the previous 10 weeks, we will not predict them
        pred_list.append(row['week_close'])
        labels3.append(row['colour'])
    else:
        X = np.array(range(index - 9, index + 1)).reshape(10, 1)
        Y = year2_data.drop(['week_open', 'colour'], axis=1).iloc[index - 10:index].reset_index(drop=True)
        poly_reg = PolynomialFeatures(degree=3)
        X_poly = poly_reg.fit_transform(X)
        model = LinearRegression().fit(X_poly, Y)
        pred_value = model.predict(poly_reg.fit_transform([[index]]))
        pred_list.append(pred_value[0][0])
        if pred_value[0][0] > year2_data['week_close'][index - 1]:
            labels3.append('green')
        if pred_value[0][0] < year2_data['week_close'][index - 1]:
            labels3.append('red')
        if pred_value[0][0] == year2_data['week_close'][index - 1]:
            labels3.append(labels3[index - 1])

print('Degree:{}\tW:{}'.format(3, 10))
acc = output_result(list(year2_data['colour'])[10:], labels3[10:])

# 4. implement three trading strategies for year 2 (for each d...
year2_data['pred_label1'] = labels1
year2_data['pred_label2'] = labels2
year2_data['pred_label3'] = labels3

all_money = 100
position = '0'
share = 0

for index, row in year2_data.iterrows():
    if row['pred_label1'] == 'green':
        if position == '0':
            share = all_money / row['week_open']
            all_money = 0
            position = '1'
        elif position == '1':
            pass

    if row['pred_label1'] == 'red':
        if position == '0':
            pass
        elif position == '1':
            all_money = share * row['week_open']
            share = 0
            position = '0'

if position == '0':
    print('Implement the trading strategy based on labels for year2(d=1,w=5), finally have:', all_money)
elif position == '1':
    all_money = share * list(year2_data['week_close'])[-1]
    print('Implement the trading strategy based on labels for year2(d=1,w=5), finally have:', all_money)

all_money = 100
position = '0'
share = 0

for index, row in year2_data.iterrows():
    if row['pred_label2'] == 'green':
        if position == '0':
            share = all_money / row['week_open']
            all_money = 0
            position = '1'
        elif position == '1':
            pass

    if row['pred_label2'] == 'red':
        if position == '0':
            pass
        elif position == '1':
            all_money = share * row['week_open']
            share = 0
            position = '0'

if position == '0':
    print('Implement the trading strategy based on labels for year2(d=2,w=8), finally have:', all_money)
elif position == '1':
    all_money = share * list(year2_data['week_close'])[-1]
    print('Implement the trading strategy based on labels for year2(d=2,w=8), finally have:', all_money)

all_money = 100
position = '0'
share = 0

for index, row in year2_data.iterrows():
    if row['pred_label3'] == 'green':
        if position == '0':
            share = all_money / row['week_open']
            all_money = 0
            position = '1'
        elif position == '1':
            pass

    if row['pred_label3'] == 'red':
        if position == '0':
            pass
        elif position == '1':
            all_money = share * row['week_open']
            share = 0
            position = '0'

if position == '0':
    print('Implement the trading strategy based on labels for year2(d=3,w=10), finally have:', all_money)
elif position == '1':
    all_money = share * list(year2_data['week_close'])[-1]
    print('Implement the trading strategy based on labels for year2(d=3,w=10), finally have:', all_money)



