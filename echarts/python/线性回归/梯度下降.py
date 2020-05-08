#coding=gbk
import numpy as np
import matplotlib.pyplot as plt
#定义一个函数进行求导，theta求导数
def dJ(theta):
    return 2*(theta-2.5)
#原函数
def J(theta):
    try:
        return (theta-2.5)**2-1
    except:
        return float('inf')
'''
梯度下降  赋予一个初值0  学习率0.1  再接着做循环迭代
'''
def gradient_descent(theta_history,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
    theta=initial_theta
    theta_history.append(theta)
    i_ters=0
    while i_ters<n_iters:
        gradient=dJ(theta)  #梯度
        last_theta=theta   #最终的值

        theta=theta-eta*gradient
        theta_history.append(theta)   #每次迭代的值放入库中
        if(abs(J(theta)-J(last_theta))<epsilon):
            break
        i_ters+=1
    return theta,theta_history
def plot_theta_history(plot_x,theta_history):
    plt.plot(plot_x,J(plot_x))
    plt.show()
    plt.plot(np.array(theta_history),J(np.array(theta_history)),color='red',marker='*')
    plt.show()

if __name__ == '__main__':
    plot_x=np.linspace(-1,6,141)
    plot_y=(plot_x-2.5)**2-1
    plt.plot(plot_x,plot_y)
    plt.show()

    theta=0.0
    eta=1.01
    epsilon=1e-8
    theta_history=[]
    gradient_descent(theta_history,theta,eta,n_iters=1e4,epsilon=1e-8)
    plot_theta_history(plot_x,theta_history)