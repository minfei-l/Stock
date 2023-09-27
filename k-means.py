import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


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


year1_weeks = pd.read_csv('year1_weeks.csv',encoding='utf-8')
year1_features = year1_weeks[['avg_daily_return','avg_volatility']]

ss = StandardScaler()
ss_year1_features = ss.fit_transform(year1_features)

#print(year1_features)
#print(ss_year1_features)

k_list = [1,2,3,4,5,6,7,8]
models = []

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?highlight=kmeans#sklearn.cluster.KMeans
distortions = []
for k in k_list:
	model = KMeans(init='random',n_clusters=k)
	kmeans = model.fit(ss_year1_features)
	models.append(model)
	distortions.append(kmeans.inertia_)
	print('K =',k,',distortion =',kmeans.inertia_)

plt.plot(k_list,distortions)
plt.show()


cluster_results = models[3].labels_
print(cluster_results)

for cluster_num in [0,1,2,3]:
	indexs = []
	colours = []
	for i in range(len(cluster_results)):
		if cluster_results[i] == cluster_num:
			indexs.append(i)
			colours.append(year1_weeks['colour'][i])

	red_per = colours.count('red')/len(colours)
	green_per = colours.count('green')/len(colours)

	print('For cluster No.{},the percentage of "green" is {}, the figure for "red" is {}'.format(cluster_num,green_per,red_per))





