import numpy as np
def file2matrix(filename):
    fr=open(filename)
    array0Lines=fr.readlines()
    print("array0Lines",array0Lines)
    number0fLines=len(array0Lines)
    retuenMat=np.zeros((number0fLines,3))
    print("returnMat",retuenMat)
    classLabelIVector=[]
    index=0
    for line in array0Lines:
        line=line.strip()
        print("line",line)
        listFromLine=line.split('\t')
        retuenMat[index,:]=listFromLine[0:3]
        print("returnMat",retuenMat)
        if listFromLine[-1]=='猪队友':
            classLabelIVector.append(1)
        elif listFromLine[-1]=='一般般':
            classLabelIVector.append(2)
        elif listFromLine[-1]=='神队友':
            classLabelIVector.append(3)
        index+=1
    return retuenMat,classLabelIVector
if __name__ =='__main__':
    filename="datingTestSet01.txt"
    datingDataMat,datingLabels=file2matrix(filename)
    print(datingDataMat)
    print(datingLabels)