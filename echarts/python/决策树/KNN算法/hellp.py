import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOflines = len(arrayOlines)
    returnMat = np.zeros((numberOflines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == '猪队友':
            classLabelVector.append(1)
        elif listFromLine[-1] == '一般般':
            classLabelVector.append(2)
        elif listFromLine[-1] == '神队友':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector
def showdatas(datingDataMat,datingLabels):
    font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('blue')  # blueblack
        if i == 2:
            LabelsColors.append('green')  # yellow  orange
        if i == 3:
            LabelsColors.append('red')  # green
    # 画出散点图,以datingDataMat矩阵的第一()、第二列()数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'抢位置与挂机占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年抢位置程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'挂机占比占', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(抢位置)、第三列(直播)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年抢位置与主播概率数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年抢位置数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'主播概率数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二()、第三列()数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'挂机占比与主播概率数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'挂机占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'主播概率数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='blue', marker='.',
                              markersize=6, label='Pig mate')
    smallDoses = mlines.Line2D([], [], color='green', marker='.',
                               markersize=6, label=u'General mate')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label=u'God mate')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()
if __name__ == '__main__':
    filename= "datingTestSet.txt"
    datingDataMat,datingLabels=file2matrix(filename)
    showdatas(datingDataMat,datingLabels)