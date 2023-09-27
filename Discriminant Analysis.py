import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
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

model1 = LinearDiscriminantAnalysis()
model2 = QuadraticDiscriminantAnalysis()
model1.fit(year1_features,year1_labels)
model2.fit(year1_features,year1_labels)

print('The equation for Linear Discriminant Analysis classifier is: \nlog P(y=k|x) = {} * 𝜇 + {} * 𝜎 + {}'.format(model1.coef_[0][0],model1.coef_[0][1],model1.intercept_[0]))

year2_weeks = pd.read_csv('year2_weeks.csv',encoding='utf-8')
year2_weeks['label'] = year2_weeks['colour'].apply(transform_color)

year2_features = year2_weeks[['avg_daily_return','avg_volatility']]
year2_labels = year2_weeks['label']

pred_lst1 = model1.predict(year2_features)
pred_lst2 = model2.predict(year2_features)

print('Linear Discriminant Analysis:')
output_result(year2_weeks['label'].to_list(),pred_lst1)
print('Quadratic Discriminant Analysis:')
output_result(year2_weeks['label'].to_list(),pred_lst2)


year2_weeks['pred_label1'] = pred_lst1

all_money = 100
position = '0'
share = 0

for index,row in year2_weeks.iterrows():
	if row['pred_label1'] == 0:
		if position == '0':
			share = all_money/row['week_open']
			all_money = 0
			position = '1'
		elif position == '1':
			pass

	if row['pred_label1'] == 1:
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
	print('Implement the trading strategy based on labels in linear model for year2, finally have:',all_money)

print('Implement the "buy-and-hold" strategy, finally have:',(100/list(year2_weeks['week_open'])[0])*list(year2_weeks['week_close'])[-1])



year2_weeks['pred_label2'] = pred_lst2

all_money = 100
position = '0'
share = 0

for index,row in year2_weeks.iterrows():
	if row['pred_label2'] == 0:
		if position == '0':
			share = all_money/row['week_open']
			all_money = 0
			position = '1'
		elif position == '1':
			pass

	if row['pred_label2'] == 1:
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
	print('Implement the trading strategy based on labels in quadratic model for year2, finally have:',all_money)


