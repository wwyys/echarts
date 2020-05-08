def fn(n):
    print("现在是第",n,"层递归")
    if n>=3:
        return
    fn(n+1)
    print("递归的第",n,"层结束")
if __name__ == '__main__':
    fn(1)
    print("程序结束，回到主程序")