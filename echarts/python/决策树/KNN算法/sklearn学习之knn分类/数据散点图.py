# ================================================
# KNN模型分类
# （身高、体重）数据，绘制数据散点图
# 2019-02-24
# ================================================

# "np" and "plt" are common aliases for NumPy and Matplotlib, respectively.
import numpy as np
import matplotlib.pyplot as plt

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

plt.figure()
plt.title('Human Heights and Weights by Sex')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', marker='x' if y_train[i] == 'male' else 'D')
plt.grid(True)
plt.show()

x=np.array([[155,70]])
print(x)
print(X_train)
distances=np.sqrt(np.sum((X_train-x)**2,axis=1))
distances