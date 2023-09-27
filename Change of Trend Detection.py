import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from scipy.stats import f as fisher_f


def SSE(true_list, pred_list):
    sse = 0
    if len(true_list) == len(pred_list):
        for i in range(len(true_list)):
            sse = sse + np.square(true_list[i] - pred_list[i])
        return sse
    else:
        print('SSE ERROR')


def find_candidate_k(k, df):
    X1 = np.array(range(1, k + 1)).reshape(k, 1)
    Y1 = df.drop(['Date', 'Month'], axis=1).iloc[:k].reset_index(drop=True)
    # print(X1)
    # print(np.array(Y1))
    lr1 = LinearRegression().fit(X1, Y1)
    # print(lr1.coef_)
    pred1 = lr1.predict(X1)
    # print(list(Y1['Adj Close']),[i[0] for i in pred1.tolist()])
    sse1 = SSE(list(Y1['Adj Close']), [i[0] for i in pred1.tolist()])

    X2 = np.array(range(k + 1, df.shape[0] + 1)).reshape(df.shape[0] - k, 1)
    Y2 = df.drop(['Date', 'Month'], axis=1).iloc[k:].reset_index(drop=True)
    # print(X2,Y2)
    lr2 = LinearRegression().fit(X2, Y2)
    pred2 = lr2.predict(X2)
    sse2 = SSE(list(Y2['Adj Close']), [i[0] for i in pred2.tolist()])

    # print('The SSE for period1(day1-{}) is {}, for period2(day{}-day{}) is {}'.format(k,sse1,k+1,df.shape[0],sse2))
    # print('-----------------------------------------------------------------------------------------------------')
    return sse1, sse2


def calculate_p_score(k, k_sse1, k_sse2, df):
    X = np.array(range(1, df.shape[0] + 1)).reshape(df.shape[0], 1)
    Y = df.drop(['Date', 'Month'], axis=1).reset_index(drop=True)
    lr = LinearRegression().fit(X, Y)
    pred = lr.predict(X)
    sse = SSE(list(Y['Adj Close']), [i[0] for i in pred.tolist()])

    f_stat = ((sse - k_sse1 - k_sse2) * (df.shape[0] - 4)) / (2 * (k_sse1 + k_sse2))
    p_value = fisher_f.cdf(f_stat, 2, df.shape[0] - 4)
    print('The P-value is ', p_value)
    print('-----------------------------------------------------------------------------------------------------')

    return p_value


year1_data = pd.read_csv('year1.csv', encoding='utf-8')[['Date', 'Month', 'Adj Close']]

for month_data in year1_data.groupby('Month'):
    # print(month_data)
    print('Month:', month_data[0])
    sse1_lst = []
    sse2_lst = []
    sse_lst = []
    for i in range(2, month_data[1].shape[0] - 2):
        # print('When k is {} (that is {})'.format(i,list(month_data[1]['Date'])[i]))
        sse1, sse2 = find_candidate_k(i, month_data[1])
        sse1_lst.append(sse1)
        sse2_lst.append(sse2)
        sse_lst.append(sse1 + sse2)

    opt_k = sse_lst.index(min(sse_lst)) + 3
    print('The candidate day (k) is:', opt_k)

    p_score = calculate_p_score(opt_k, sse1_lst[opt_k - 3], sse2_lst[opt_k - 3], month_data[1])

print('\n')

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

year2_data = pd.read_csv('year2.csv', encoding='utf-8')[['Date', 'Month', 'Adj Close']]

for month_data in year2_data.groupby('Month'):
    print('Month:', month_data[0])
    sse1_lst = []
    sse2_lst = []
    sse_lst = []
    for i in range(3, month_data[1].shape[0] - 2):
        # print('When k is {} (that is {})'.format(i,list(month_data[1]['Date'])[i]))
        sse1, sse2 = find_candidate_k(i, month_data[1])
        sse1_lst.append(sse1)
        sse2_lst.append(sse2)
        sse_lst.append(sse1 + sse2)

    opt_k = sse_lst.index(min(sse_lst)) + 3
    print('The candidate day (k) is:', opt_k)

    calculate_p_score(opt_k, sse1_lst[opt_k - 3], sse2_lst[opt_k - 3], month_data[1])

