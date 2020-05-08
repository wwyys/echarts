from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
from sklearn import model_selection
iris=datasets.load_iris()
print(iris)
#打乱数据
data_size=iris.data.shape[0]
index=[i for i in range(data_size)]
random.shuffle(index)
iris.data=iris.data[index]
iris.target=iris.target[index]
#切分数据
test_size=40
x_train=iris.data[test_size:]
x_test=iris.data[:test_size]
y_train=iris.target[test_size:]
y_test=iris.target[:test_size]
#构建模型
model=neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
prediction=model.predict(x_test)

print(classification_report(y_test,prediction))