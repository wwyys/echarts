from pyecharts import options as opts
from pyecharts.charts import Liquid,Page
from pyecharts.globals import SymbolType
def liquid() -> Liquid:
    c=(
        Liquid()
        .add('lq',[0.67,0.30,0.15])
        .set_global_opts(title_opts=opts.TitleOpts(title="Liquid"))
    )
    return c
liquid().render('C:/Users/DF/Desktop/data/liquid.html')