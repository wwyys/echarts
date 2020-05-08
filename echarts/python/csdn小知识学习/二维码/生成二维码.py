'''
项目：生成二维码
项目人：zxl
日期：  2020.04.03
介绍：
法1生成黑白的二维码
法2生成有背景的二维码
'''
'''
#法1
from MyQR import myqr
myqr.run(words='http://www.baidu.com')  #如果为网站则会自动跳转，文本直接显示，不支持中文
'''

#法2
from MyQR import myqr
myqr.run(
    words='http://wwww.baidu.com',  #包含信息
    picture='timg.jpg',     #背景图片
    colorized=True,    #是否有颜色，如果False为黑白色
    save_name='code1.png'        #输出文件名
)