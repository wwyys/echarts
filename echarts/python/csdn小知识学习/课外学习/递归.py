def fn(n):
    print("�����ǵ�",n,"��ݹ�")
    if n>=3:
        return
    fn(n+1)
    print("�ݹ�ĵ�",n,"�����")
if __name__ == '__main__':
    fn(1)
    print("����������ص�������")