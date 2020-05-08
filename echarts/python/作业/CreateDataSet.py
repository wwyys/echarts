

import numpy as np
x=np.random.random(10)
y=np.random.random(10)
X=np.vstack([x,y])
sk=np.var(X,axis=0,ddof=1)
d1=np.sqrt(((x-y)**2/sk).sum())
print(d1)
def createDataSet():
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels
if __name__=='__main__':
    group,labels=createDataSet()
    print("数据样本group:",group)
    print("特征标签labels:",labels)
