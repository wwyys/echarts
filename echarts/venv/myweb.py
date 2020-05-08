#coding:utf-8
__author__ = 'zxl'
__date__ = '2020/04/21'
from flask_bootstrap import Bootstrap
from flask import Flask,render_template
app=Flask(__name__)
bootstrap = Bootstrap(app)
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/js')
def jstest():
    return render_template('jstest.html')
'''
@app.route('/css')
def csstest():
    return render_template('csstest.html')
'''
if __name__ == '__main__':
    app.run(debug="debug")

