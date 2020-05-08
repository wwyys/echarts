import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
铁三角：
模型
损失函数 cost function
优化方法
'''


'''
功能：模型
input:自变量X
outout:应变量Y
'''
def model(a,b,x):
    return a*x+b
'''
功能：损失函数  5个数据
input:自变量x
output：最小误差值
'''
def cost_function(a,b,x,y):
    n=5
    return 0.5/n*(np.square(y-a*x-b)).sum()
'''
功能：优化方法  5个数据
input:自变量x
output：A,B
'''
def optimize(a,b,x,y):
    n=5
    alpha=1e-1
    y_hat=model(a,b,x)   #预测值
    da=(1.0/n)*((y_hat-y).sum())   #对a求偏导
    print("da=",da)
    db=(1.0/n)*((y_hat-y).sum())
    print("db=",da)
    a=a-alpha*da
    b=b-alpha*db
    return a,b
'''
功能：迭代5个数据
input：times迭代次数
output：A,B
'''
def iterate(a,b,x,y,times):   #times迭代次数
    for i in range(times):
        a,b=optimize(a,b,x,y)
    y_hat=model(a,b,x)
    cost=cost_function(a,b,x,y)
    print(a,b,cost)
    plt.scatter(x,y)
    plt.plot(x,y_hat)
    plt.show()
    return a,b
'''
功能：模型评价计算R平方
input：a，b，x

'''


if __name__ == '__main__':
    x=[13854,12213,11009,10655,9503]    #程序员工资
    x=np.reshape(x,newshape=(5,1))/10000.0
    y=[21332,20162,19138,18621,18016]    #算法工程师
    y=np.reshape(y,newshape=(5,1))/10000.0
    plt.scatter(x,y)
    plt.show()
    #初始化
    a=0.0
    b=0.0
    times=1000
    a,b=iterate(a,b,x,y,times)
    SST,SSR,SSE,
