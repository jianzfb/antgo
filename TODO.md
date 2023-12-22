* 有名字的管线
with pipeline() as xxx
    建立管线

* 注册算子组
with pipeline() as xxx:
    group("name"). \
        placeholder['a'](). \
        placeholder['b'](). \
        op[]()

    glob[](). \
        group.name[('a', 'b'), 'out']().run()


# python 端
d = pipeline.group("name")(a,b,c)

# c++ 端



