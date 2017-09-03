# coding=utf-8

import py_sub as sub


if __name__ == '__main__':
    sub.add_float('float', 1.)
    sub.add_int('int', 5)
    sub.add_str('string', 'new-string')
    sub.print_all()

    # 输出：
    # int:5
    # float:1.0
    # string:new-string