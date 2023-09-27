import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score


def output_result(y_true,y_pred):
	cm = confusion_matrix(y_true,y_pred)#TP FN FP TN
	tpr = cm[0][0]/(cm[0][0]+cm[0][1])
	tnr = cm[1][1]/(cm[1][1]+cm[1][0])
	accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
	print('Confusion Matrics: ',cm)
	print('Accuracy Score:',accuracy)
	print('TPR: ',tpr)
	print('TNR: ',tnr)
	print('----------------------------------------------------------------')
	return accuracy


def transform_color(s):
	if s == 'green':
		return 0
	elif s == 'red':
		return 1


year1_weeks = pd.read_csv('year1_weeks.csv',encoding='utf-8')
year1_weeks['label'] = year1_weeks['colour'].apply(transform_color)

year1_features = year1_weeks[['avg_daily_return','avg_volatility']]
year1_labels = year1_weeks['label']

model = GaussianNB()
model.fit(year1_features,year1_labels)
score = model.score(year1_features,year1_labels)
print('Accuracy Score of Gaussian naive bayesian classifier for year 1 is:',score)

#implement a Gaussian naive bayesian classifier and compute its accuracy for year 2
year2_weeks = pd.read_csv('year2_weeks.csv',encoding='utf-8')
year2_weeks['label'] = year2_weeks['colour'].apply(transform_color)

year2_features = year2_weeks[['avg_daily_return','avg_volatility']]
year2_labels = year2_weeks['label']

pred_lst = model.predict(year2_features)
output_result(year2_weeks['label'].to_list(),pred_lst)

# implement a trading strategy based on your labels for year 2 and compare the performance with the ”buy-and-hold” strategy. Which strategy results in a larger amount at the end of the year?
year2_weeks['pred_label'] = pred_lst

all_money = 100
position = '0'
share = 0

for index,row in year2_weeks.iterrows():
	if row['pred_label'] == 0:
		if position == '0':
			share = all_money/row['week_open']
			all_money = 0
			position = '1'
		elif position == '1':
			pass

	if row['pred_label'] == 1:
		if position == '0':
			pass
		elif position == '1':
			all_money = share * row['week_open']
			share = 0
			position = '0'

if position == '0':
	print('finally have:',all_money)
elif position == '1':
	all_money = share * list(year2_weeks['week_close'])[-1]
	print('Implement the trading strategy based on labels for year2, finally have:',all_money)

print('Implement the "buy-and-hold" strategy, finally have:',(100/list(year2_weeks['week_open'])[0])*list(year2_weeks['week_close'])[-1])




