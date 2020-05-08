# ================================================
# KNN模型分类，sklearn类库实现
# （身高、体重）数据，预测
# 2019-02-24
# ================================================
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])
y_train = ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female']
lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)
print(y_train_binarized)  #1为男性


K = 3
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized.reshape(-1))
prediction_binarized = clf.predict(np.array([155, 70]).reshape(1, -1))[0]
predicted_label = lb.inverse_transform(prediction_binarized)
print(predicted_label)
# ================================================
# KNN模型分类，sklearn类库实现
# 测试集进行预测效果分析
# 2019-02-24
# ================================================

X_test = np.array([
    [168, 65],
    [180, 96],
    [160, 52],
    [169, 67],
    [178, 64],
    [172, 59]
])
y_test = ['male', 'male', 'female', 'female','male','female']
y_test_binarized = lb.transform(y_test)
print('Binarized labels: %s' % y_test_binarized.T[0])

predictions_binarized = clf.predict(X_test)
print('Binarized predictions: %s' % predictions_binarized)
print('Predicted labels: %s' % lb.inverse_transform(predictions_binarized))
print('=======计算正确率=====')
from sklearn.metrics import accuracy_score
print('Accuracy: %s' % accuracy_score(y_test_binarized, predictions_binarized))
print('========计算召回率=======')
from sklearn.metrics import recall_score
print('Recall: %s' % recall_score(y_test_binarized, predictions_binarized))

print('========计算F1分数=======')
from sklearn.metrics import f1_score
print('F1 score: %s' % f1_score(y_test_binarized, predictions_binarized))

print('========生成综合报告=======')
from sklearn.metrics import classification_report
print(classification_report(y_test_binarized, predictions_binarized, target_names=['male'], labels=[1]))