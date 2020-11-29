import torch as t
import visdom


vis = visdom.Visdom(env=u'test1',use_incoming_socket=False)
# x = t.arange(1,10,0.01)
# y = t.sin(x)
# y1 = t.cos(x)
# vis.line(X=x,Y=y,win='sin&cos',update='None')
# vis.line(X=x,Y=y1,win='sinx',name='new',update='append')

vis.text(u'''<h1>Hello Visdom</h1><br>Visdom是Facebook专门为<b>PyTorch</b>开发的一个可视化工具，
         在内部使用了很久，在2017年3月份开源了它。

         Visdom十分轻量级，但是却有十分强大的功能，支持几乎所有的科学运算可视化任务''',
         win='visdom',
         opts={'title': u'visdom简介'}
         )