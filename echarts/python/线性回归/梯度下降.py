#coding=gbk
import numpy as np
import matplotlib.pyplot as plt
#����һ�����������󵼣�theta����
def dJ(theta):
    return 2*(theta-2.5)
#ԭ����
def J(theta):
    try:
        return (theta-2.5)**2-1
    except:
        return float('inf')
'''
�ݶ��½�  ����һ����ֵ0  ѧϰ��0.1  �ٽ�����ѭ������
'''
def gradient_descent(theta_history,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
    theta=initial_theta
    theta_history.append(theta)
    i_ters=0
    while i_ters<n_iters:
        gradient=dJ(theta)  #�ݶ�
        last_theta=theta   #���յ�ֵ

        theta=theta-eta*gradient
        theta_history.append(theta)   #ÿ�ε�����ֵ�������
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