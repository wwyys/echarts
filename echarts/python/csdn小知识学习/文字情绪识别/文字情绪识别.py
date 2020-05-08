import paddlehub as hub
senta=hub.Module(name='senta_lstm')   #加载模型
sentence=[#准备要识别的语句
          '','','','','','',
        ]
results=senta.sentiment_classify(data={"text":sentence})   #情绪识别
for result in results:
    print(result)