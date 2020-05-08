N=int(input())
list=[]
for i in range(2,N):
    for p in range(2,i):
        if (i%p)==0:
            break
        else:
            list.append(i)
            print(max(list))