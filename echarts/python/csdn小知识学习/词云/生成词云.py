from wordcloud import WordCloud
wc=WordCloud()    #创建词云对象
wc.generate('Since I lost you, my world is lack of light.')

wc.to_file('wc.png')
