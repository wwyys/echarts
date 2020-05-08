from pyecharts import charts
gauge=charts.Gauge()
gauge.add('Python小例子',[('Python机器学习',30),('Python基础',70),('Python正则',90)])
gauge.render(path="C:/Users/DF/Desktop/data/仪表盘.html")
print('ok')