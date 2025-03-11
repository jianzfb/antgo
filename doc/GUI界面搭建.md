# GUI快速界面入门
## 基础知识

目前仅支持tkinter UI引擎。

## 基础架构

基于管线架构，实现UI搭建。基于双向绑定数据，实现模块间数据联动。使用gui.tk启用tkinter UI引擎，并同时设置窗口长宽和名字。后续通过ui.tk来调用tkinter UI引擎的控件。通过[]符号，进行关联主从关系。

## 例子


```
gui.tk['root'](title='CSI波形显示', height=500, width=400). \
    ui.smart.data[['magnitude', 'phase']](source=CSIData). \
    ui.tk.layout['root', 'top_frame'](width=1.0, height=0.5, bg='white', gridy=0, gridx=0, spanx=10, padx=0, pady=5). \
    ui.tk.layout['root', 'bottom_frame'](width=1.0, height=0.5, bg='white', gridy=1, gridx=0, spanx=10, padx=0, pady=5). \
    ui.tk.canvas.line[('top_frame', 'magnitude'), 'magnitude_canvas'](fill='blue'). \
    ui.tk.canvas.line[('bottom_frame', 'phase'), 'phase_canvas'](fill='red').loop()

```
产生布局效果
![](https://image.vibstring.com/2761741708136_.pic.jpg)