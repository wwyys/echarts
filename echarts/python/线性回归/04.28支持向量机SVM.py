# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns;sns.set()

#创建模拟数据
from sklearn.datasets.samples_generator import make_blobs
x,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.60)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='autumn')
plt.show()

#可以用多种方法分类
xfit=np.linspace(-1,3.5)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='autumn')
for a,b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit,a*xfit+b,'-k')
plt.xlim(-1,3.5)
plt.show()

#SVM:假设每一条分割线是有宽度的
xfit=np.linspace(-1,3.5)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='autumn')
for a,b,d in [(1,0.65,0.33),(0.5,1.6,0.55),(-0.2,2.9,0.2),(-0.2,2.9,0.2)]:
    yfit=a*xfit+b
    plt.plot(xfit,yfit,'-k')

    plt.fill_between(xfit,yfit-d,yfit+d,edgecolor='none',color='#AAAAAA',alpha=0.4)
plt.xlim(-1,3.5)
plt.show()
#在SVM的框架下，认为最宽的线为最优的分割线
#训练SVM
#使用线性SVM和比较大的C
from sklearn.svm import SVC
model = SVC(kernel='linear',C=1E10)
model.fit(x,y)
#创建一个显示SVM分割线的函数
def plot_svc_decision_function(model,ax=None,plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    #create grid to evaluate model
    x=np.linspace(xlim[0],xlim[1],30)
    y=np.linspace(ylim[0],ylim[1],30)
    Y,X=np.meshgrid(y,x)
    xy=np.vstack([X.ravel(),Y.ravel()]).T
    P=model.decision_function(xy).reshape(X.shape)
    #plot decision boundary and margins
    ax.contour(X,Y,P,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
    #plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],
                   model.support_vectors_[:,1],
                   s=300,linewidth=1,facecolors='none');
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='autumn')
plot_svc_decision_function(model);
plt.show()
#非支持向量的数据，对分割线没有影响
#只有支持向量会影响分割线，如果我们添加一些非支持向量的数据，对分割线没有影响
def plot_svm(N=10,ax=None):
    x,y=make_blobs(n_samples=200,centers=2,random_state=0,cluster_std=0.60)
    x=x[:N]
    y=y[:N]
    model=SVC(kernel='linear',C=1E10)
    model.fit(x,y)
    ax=ax or plt.gca()
    ax.scatter(x[:,0],x[:,1],c=y,s=50,cmap='autumn')
    ax.set_xlim(-1,4)
    ax.set_ylim(-1,6)
    plot_svc_decision_function(model,ax)

fig,ax=plt.subplots(1,2,figsize=(16,6))
fig.subplots_adjust(left=0.0625,right=0.95,wspace=0.1)
for axi,N in zip(ax,[60,120]):
    plot_svm(N,axi)
    axi.set_title('N={0}'.format(N))
plt.show()