"""
读取数据
Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签
"""
def loadDataSet(fileName):
    dataMat=[];
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():   #逐行读取，清除空格
        lineArr =line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])  #添加数据
        labelMat.append(float(lineArr[2]))    #添加标签
    return dataMat,labelMat
"""
计算误差  #格式化计算误差的函数，方便多次调用
Parameters:
oS- 数据结构
k- 标号为k的数据
Returns：
Ek - 标号为k的数据误差
"""
def calcEk(oS,k):
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)+oS.b)
    Ek=fXk-float(oS.labelMat[k])
    return Ek
"""
函数说明：随机选择alpha_j的索引值
#修改选择第二个变量alphaj的方法
Parameters:
i-alpha_i的索引值
m-alpha参数个数
Returns：
J-aplpha_j的索引值
"""
def selectJrand(i,m):
    j=i    #选择一个不等于i的j
    while (j==i):
        j=int(random.uniform(0,m))
    return j
"""
内循环启发方式2  #修改选择第二个变量alphaj的方法
Parameters:
i - 编号为i的数据的索引范围
oS - 数据结构
Ei - 标号为i的数据误差
Returns:
j,maxK -  编号为j或maxK的数据的索引值
Ej -  编号为j的数据误差
"""
def selectJ(i,oS,Ei):
    maxK =-1;
    maxDeltaE =0;
    Ej=0     #初始化
    oS.eCache[i] = [1,Ei]   #要根据Ei更新误差缓冲
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]   #返回误差不为0的数据的索引值
    if (len(validEcacheList))>1:   #有不为0的误差
        for k in validEcacheList:    #遍历，找到最大的Ek
            if k==i:              #不计算i,浪费时间
                continue
            Ek = calcEk(oS,k)     #计算Ek
            delaE=abs(Ei-Ek)     #计算Ei-Ek
            if (delaE>maxDeltaE):    #找到maxDeltaE
                maxK=k; maxDeltaE=delaE;Ej=Ek
        return maxK,Ej          #返回maxK,Ej
    else:                      #没有不为0的误差
        j=selectJrand(i.oS.m)   #随机选择alpha_j的索引值
        Ej=calcEk(oS,j)         #计算Ej
    return j,Ej                 #j,Ej
"""
数据结构，维护所有需要操作的值
Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
"""
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn    #数据矩阵
        self.labelMat = classLabels   #数据标签
        self.C = C   #松弛变量
        self.tol = toler   #容错率
        self.m = np.shape(dataMatIn)[0]    #数据矩阵行数
        self.alphas = np.mat(np.zeros((self.m,1)))   #根据矩阵行数初始化alpha参数为0
        self.b = 0    #初始化b参数为0
        self.eCache = np.mat(np.zeros((self.m,2)))   #根据矩阵行数初始化误差缓存，第一列为
"""
计算Ek,并更新误差缓冲
Parameters:
oS  - 数据结构
k - 标号为k的数据的索引值
Returns:
无
"""
def updateEk(oS,k):
    Ek=calcEk(oS,k)       #计算Ek
    oS.eCache[k]=[1,Ek]    #更新误差缓冲
"""
修剪alpha_j
Parameters:
aj_alpha_j
H-alpha
L-alpha
"""
