from datetime import datetime
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_night_data(filename):
    stock_data = pd.read_csv(filename + '.csv', encoding='utf-8')

    nights = []
    pre_night_opens = []
    pre_night_closes = []
    post_night_opens = []
    positions = [] # trading method
    profits = []

    all_money = 100
    values = [100] # first value is 100

    for index, row in stock_data[:-1].iterrows(): ### the last day stop trading
        nights.append(row['Date']) #the day that trade
        pre_night_opens.append(row['Open']) #open price
        pre_night_closes.append(row['Close']) #closed price
        post_night_opens.append(stock_data['Open'][index + 1]) ### ?
        if row['Close'] > row['Open']:
            positions.append('long')
            night_profit = round(all_money / row['Close'] * (stock_data['Open'][index + 1] - row['Close']), 2) ### ?
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

        values.append(all_money) ### save the on-hand money on that day

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

    # On the same plot, show the growth of your portfolio for your stock and SPY and buy-and-hold strategy
    bh_share = round(100 / stock_data['Open'][0], 2) # how many stock to buy on the first day
    bh_value_lst = []
    for index, row in stock_data.iterrows():
        bh_value_lst.append(bh_share * row['Close']) # how much per day

    return values, bh_value_lst


def get_night_data_with_restriction(filename, thsd):
    stock_data = pd.read_csv(filename + '.csv', encoding='utf-8')

    nights = []
    pre_night_opens = []
    pre_night_closes = []
    post_night_opens = []
    positions = []
    profits = []

    all_money = 100

    for index, row in stock_data[:-1].iterrows():
        nights.append(row['Date'])
        pre_night_opens.append(row['Open'])
        pre_night_closes.append(row['Close'])
        post_night_opens.append(stock_data['Open'][index + 1])

        rise_or_fall = (row['Close'] - row['Open']) / row['Open'] #涨幅

        if rise_or_fall > thsd:
            positions.append('long')
            night_profit = round(all_money / row['Close'] * (stock_data['Open'][index + 1] - row['Close']), 2)
            all_money = all_money + night_profit
            profits.append(night_profit)
        elif rise_or_fall < 0 - thsd:
            positions.append('short')
            night_profit = round(all_money / row['Close'] * (row['Close'] - stock_data['Open'][index + 1]), 2)
            all_money = all_money + night_profit
            profits.append(night_profit)
        else:
            positions.append('no action')
            profits.append(0)

    night_data = pd.DataFrame(
        columns=['night', 'pre_night_open', 'pre_night_close', 'post_night_open', 'position', 'profit'])
    night_data['night'] = nights
    night_data['pre_night_open'] = pre_night_opens
    night_data['pre_night_close'] = pre_night_closes
    night_data['post_night_open'] = post_night_opens
    night_data['position'] = positions
    night_data['profit'] = profits

    # print(night_data)

    avg_profit_per_trade = night_data[night_data['position'] != 'no action']['profit'].mean()
    avg_profit_per_long_position = night_data[night_data['position'] == 'long']['profit'].mean()
    avg_profit_per_short_position = night_data[night_data['position'] == 'short']['profit'].mean()

    return (avg_profit_per_trade, avg_profit_per_long_position, avg_profit_per_short_position)


def main():
    data = pd.read_csv('LNVGF.csv', encoding='utf-8')
    time_range = [datetime.strptime(d, '%Y-%m-%d').date() for d in list(data['Date'])]

    insomnia_values = []
    bh_values = []

    for name in ['LNVGF', 'SPY']:
        insomnia_value, bh_value = get_night_data(name)
        insomnia_values.append(insomnia_value)
        bh_values.append(bh_value)

        thsds = []
        avg_profit_per_trades = []
        avg_profit_per_long_positions = []
        avg_profit_per_short_positions = []

        for i in range(1, 101):
            # print(i)
            thsds.append(i / 1000)
            temp1, temp2, temp3 = get_night_data_with_restriction(name, i / 1000)
            avg_profit_per_trades.append(temp1)
            avg_profit_per_long_positions.append(temp2)
            avg_profit_per_short_positions.append(temp3)
        print(thsds, avg_profit_per_trades, avg_profit_per_long_positions, avg_profit_per_short_positions)

        plt.figure()
        plt.plot(thsds, avg_profit_per_trades)
        plt.title('Average Profit per Trade for ' + name)
        plt.show()

        plt.figure()
        plt.plot(thsds, avg_profit_per_long_positions, label='long position')
        plt.plot(thsds, avg_profit_per_short_positions, label='short position')
        plt.title('Average Profit per Trade for ' + name + '(Long/Short Positions)')
        plt.legend()
        plt.show()

    plt.figure()
    plt.plot(time_range, insomnia_values[0], label='LNVGF (Insomnia Strategy)')
    plt.plot(time_range, insomnia_values[1], label='SPY (Insomnia Strategy)')
    plt.plot(time_range, bh_values[0], label='LNVGF (Buy-and-Hold Strategy)')
    plt.plot(time_range, bh_values[1], label='SPY (Buy-and-Hold Strategy)')
    plt.title('Answer5: Growth of my portfolio for LNVGF and SPY')
    plt.legend()
    plt.show()


main()
