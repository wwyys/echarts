from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model,axis):
    x0,x1=np.meshgrid(np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
                      np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
                      )
    x_new=np.c_[x0.ravel(),x1.ravel()]
    y_pre=model.predict(x_new)
    zz=y_pre.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    cus=ListedColormap(["#EF9A9A","#FFF59D","#90CAF9"])
    plt.contourf(x0,x1,zz,cmap=cus)
#数据获取
d=datasets.load_iris()
x=d.data[:,:2] #选取的是特征数据集的前两列数据
y=d.target

print("x=",x)
print("y=",y)
#测试集和训练集
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)
#OVR方式调用    ---默认方式
log_reg=LogisticRegression()   #默认方式  没有参数
log_reg.fit(x_train,y_train)
print(log_reg.score(x_test,y_test))
plot_decision_boundary(log_reg,axis=[4,9,1,5])
plt.scatter(x[y==0,0],x[y==0,1],color="r")
plt.scatter(x[y==1,0],x[y==1,1],color="g")
plt.scatter(x[y==2,0],x[y==2,1],color="b")
plt.show()
#第二方法   OVO   效果要好于OVR  n(n-1)/2
log_reg1=LogisticRegression(multi_class="multinomial",solver="newton-cg")
log_reg1.fit(x_train,y_train)
print(log_reg1.score(x_test,y_test))
plot_decision_boundary(log_reg1,axis=[4,9,1,5])
plt.scatter(x[y==0,0],x[y==0,1],color="blue")
plt.scatter(x[y==1,0],x[y==1,1],color="orange")
plt.scatter(x[y==2,0],x[y==2,1],color="pink")
plt.show()
