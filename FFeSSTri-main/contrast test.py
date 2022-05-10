import warnings
import pandas as pd
import numpy as np
import sklearn
import scipy.io as scio
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sklearn.metrics as sm
from sklearn.svm import SVC
import sklearn.linear_model as sk_linear
from sklearn.tree import DecisionTreeClassifier
import sklearn.naive_bayes as sk_bayes
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

# model = sk_linear.LogisticRegression(penalty='l2', dual=False, C=1.0, n_jobs=1, random_state=20,
#                                      fit_intercept=True)

model = AdaBoostClassifier(n_estimators=50)
# model = sk_bayes.BernoulliNB()
# model = sk_bayes.GaussianNB()
# model = sk_bayes.GaussianNB()  # 高斯分布的朴素贝叶斯
# model = RandomForestClassifier()
indicator = ['pre', 'rec', 'f1', 'acc', 'auc']
datasets = ['cm1', 'kc2', 'kc1', 'jm1']
i=0
k=3
data = scio.loadmat("clean_data/" + datasets[k])
# X = data.iloc[:, [0, 20]]
# y = data.iloc[:, [21]]
X = data['X']
# X = X[:, [6, 1, 5, 10, 3, 19, 17, 12, 16]]
y = data['y'][0]

# smo_1 = SMOTE(random_state=0,k_neighbors=3)
# smo_2 = SMOTE(random_state=0,k_neighbors=1)
#
# try:
#     X, y =  smo_1.fit_resample(X, y)
# except ValueError:
#     X, y =  smo_2.fit_resample(X, y)
r=0.25
rate=0.3
proj_score = []
try:
    for ind in indicator:
        proj_score.append(pd.read_csv("score_1/" + str(rate) + "/" + datasets[k] + "/" + ind + ".csv", index_col=0))
except:
    for ind in indicator:
        proj_score.append(pd.DataFrame())
curr_vad = 0
for i in range(200):
    # 索引
    train_index = np.random.choice(X.shape[0], int((X.shape[0]*(1-r))*rate), replace=False)  #从第一个参数中随机抽取数字，组成指定大小的数组，replace为false时，则说明不能取相同的数字
    # list（set）是对原列表进行去重，并按照从小到大排列
    rest_index = list(set(np.arange(X.shape[0])) - set(train_index))# shape[0]输出行数
    test_index = np.random.choice(rest_index, int(X.shape[0]*r), replace=False)
    u_index = list(set(rest_index) - set(test_index))
    traindata = X[train_index]
    trainlabel = y[train_index]

    testdata = X[test_index]
    testlabel = y[test_index]

    udata = X[u_index]

    print(traindata.shape, testdata.shape, udata.shape)
    print(trainlabel.shape, testlabel.shape)
    # smo = SMOTE(random_state=0,k_neighbors=3)
    # traindata, trainlabel =  smo.fit_sample(traindata, trainlabel)


    model.fit(traindata, trainlabel)
    res1 = model.predict(testdata)
    # print(res1)
    # print(testlabel)
    algorithm = 'Tri_Training'

    proj_score[0].loc[curr_vad, algorithm] = sm.precision_score(res1, testlabel)
    proj_score[1].loc[curr_vad, algorithm] = sm.recall_score(res1, testlabel)
    proj_score[2].loc[curr_vad, algorithm] = sm.f1_score(res1, testlabel)
    proj_score[3].loc[curr_vad, algorithm] = sm.accuracy_score(res1, testlabel)
    proj_score[4].loc[curr_vad, algorithm] = sm.roc_auc_score(res1, testlabel)



    curr_vad += 1
    print('dataset:', datasets[k], '****** validation count:', curr_vad)
for i, ind in enumerate(indicator):
    proj_score[i].to_csv("score_1/"+str(rate)+"/"+"jm"+"/"+ind+".csv")
    print(ind, proj_score[i].mean().values)




