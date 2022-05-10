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

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


class TriTraining:
    def __init__(self, classifier):
        if sklearn.base.is_classifier(classifier):
            self.classifiers = [sklearn.base.clone(classifier) for i in range(3)]
        else:
            self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(3)]

    def fit(self, L_X, L_y, U_X):
        for i in range(3):
            sample = sklearn.utils.resample(L_X, L_y)
            self.classifiers[i].fit(*sample)
        e_prime = [0.5] * 3
        l_prime = [0] * 3
        e = [0] * 3
        update = [False] * 3
        Li_X, Li_y = [[]] * 3, [[]] * 3  # to save proxy labeled data
        improve = True
        self.iter = 0

        while improve:
            self.iter += 1  # count iterations

            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(L_X, L_y, j, k)
                if e[i] < e_prime[i]:
                    U_y_j = self.classifiers[j].predict(U_X)
                    U_y_k = self.classifiers[k].predict(U_X)
                    Li_X[i] = U_X[U_y_j == U_y_k]  # when two models agree on the label, save it
                    Li_y[i] = U_y_j[U_y_j == U_y_k]
                    if l_prime[i] == 0:  # no updated before
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(Li_y[i]):
                        if e[i] * len(Li_y[i]) < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            L_index = np.random.choice(len(Li_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                            Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                            update[i] = True


            for i in range(3):
                if update[i]:
                    self.classifiers[i].fit(np.append(L_X, Li_X[i], axis=0), np.append(L_y, Li_y[i], axis=0))
                    e_prime[i] = e[i]
                    l_prime[i] = len(Li_y[i])

            if update == [False] * 3:
                improve = False  # if no classifier was updated, no improvement

    def predict(self, X):
        pred = np.asarray([self.classifiers[i].predict(X) for i in range(3)])
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        return pred[0]

    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))

    def measure_error(self, X, y, j, k):
        j_pred = self.classifiers[j].predict(X)
        k_pred = self.classifiers[k].predict(X)
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
        # wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index) / sum(j_pred == k_pred)


# dataFile = 'clean_data\\cm1.mat'
# data = scio.loadmat(dataFile)
#
# traindata = np.double(data['X_train'])
# trainlabel = np.double(data['y_train'])
# testdata = np.double(data['X_test'])
# testlabel = np.double(data['X_test'])
#
# data = np.row_stack([traindata, testdata])   # 就是把训练集和测试接合并起来
# label = np.row_stack([trainlabel, testlabel]).argmax(axis=1)

# #索引
# train_index = np.random.choice(data.shape[0],100, replace=False)  #从第一个参数中随机抽取数字，组成指定大小的数组，replace为false时，则说明不能取相同的数字
# # list（set）是对原列表进行去重，并按照从小到大排列
# rest_index = list(set(np.arange(data.shape[0])) - set(train_index))# shape[0]输出行数
# test_index = np.random.choice(rest_index, 30, replace=False)
# u_index = list(set(rest_index) - set(test_index))

# traindata = data[train_index]
# trainlabel = label[train_index]
#
# testdata = data[test_index]
# testlabel = label[test_index]
#
# udata = data[u_index]

# print(traindata.shape, testdata.shape, udata.shape)
# print(trainlabel.shape, testlabel.shape)

# clf = RandomForestClassifier()
# clf.fit(traindata, trainlabel)
# res1 = clf.predict(testdata)
# print(accuracy_score(res1, testlabel))
indicator = ['pre', 'rec', 'f1', 'acc']
datasets = ['cm1', 'kc2', 'kc1', 'pc1']
k=3
data = scio.loadmat("clean_data/" + datasets[k])
i=0
X = data['X']
y = data['y'][0]
smo_1 = SMOTE(random_state=0,k_neighbors=3)
smo_2 = SMOTE(random_state=0,k_neighbors=1)

try:
    X, y =  smo_1.fit_sample(X, y)
except ValueError:
    X, y =  smo_2.fit_sample(X, y)
r=0.25
rate=0.4
proj_score = []
try:
    for ind in indicator:
        proj_score.append(pd.read_csv("score/" + str(rate) + "/" + datasets[k] + "/" + ind + ".csv", index_col=0))
except:
    for ind in indicator:
        proj_score.append(pd.DataFrame())
curr_vad = 0
for i in range(20):
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

    TT = TriTraining([RandomForestClassifier(), RandomForestClassifier(), RandomForestClassifier()])
    TT.fit(traindata, trainlabel, udata)
    res1 = TT.predict(testdata)
    # print(res1)
    # print(testlabel)
    algorithm = 'Tri_Trainingh'

    proj_score[0].loc[curr_vad, algorithm] = sm.precision_score(res1, testlabel)
    proj_score[1].loc[curr_vad, algorithm] = sm.recall_score(res1, testlabel)
    proj_score[2].loc[curr_vad, algorithm] = sm.f1_score(res1, testlabel)
    proj_score[3].loc[curr_vad, algorithm] = sm.accuracy_score(res1, testlabel)


    curr_vad += 1
    print('dataset:', datasets[k], '****** validation count:', curr_vad)
for i, ind in enumerate(indicator):
    #proj_score[i].to_csv("score/"+str(rate)+"/"+datasets[k]+"/"+ind+".csv")
    print(ind, proj_score[i].mean().values)

    # print('acc:', accuracy_score(res1, testlabel))
    # print('pre:', precision_score(res1, testlabel))
    # print('rec:', recall_score(res1, testlabel))
    # print('f1:', f1_score(res1, testlabel))



# idx = np.load("index/cross_vad/" + str(labeled_rate) + '/' + datasets[k] + '.npz')
# train_idx, test_idx, label_idx = idx['train_idx'], idx['test_idx'], idx['label_idx']




# for i in range(10):
#     train_idx_curr, test_idx_curr, label_idx_curr = train_idx[i], test_idx[i], label_idx[i]
#     for train_index, test_index, label_index in zip(train_idx_curr, test_idx_curr, label_idx_curr):
#         X_train, y_train = X[label_index], y[label_index]
#         X_test, y_test = X[test_index], y[test_index]
#         #y_train = np.ones(len(y_train_t)) * -1
#         #y_train[label_index] = y_train_t[label_index]
#         u_index = list(set(train_index) - set(label_index))
#         print(train_index)
#         print(test_index)
#         print(label_index)
#
#         TT = TriTraining([RandomForestClassifier(), RandomForestClassifier(), RandomForestClassifier()])
#         TT.fit(X_train, y_train, X[u_index])
#         res2 = TT.predict(X_test)
#
#         pre = sm.precision_score(y_test, res2)
#         re = sm.recall_score(y_test, res2)
#         f1 = sm.f1_score(y_test, res2)
#         acc = sm.accuracy_score(res2, y_test)
#         print(pre)
#         print(re)
#         print(f1)
#         print(acc)
#
#         curr_vad += 1
#         print('dataset:', datasets[k], '****** validation count:', curr_vad)



