


var_dic = {}


def add_float(name, value):
    assert isinstance(name, str)
    assert isinstance(value, float)
    var_dic[name] = value

def add_int(name, value):
    assert isinstance(name, str)
    assert isinstance(value, int)
    var_dic[name] = value

def add_str(name, value):
    assert isinstance(name, str)
    assert isinstance(value, str)
    var_dic[name] = value

def print_all():
    for (name, value) in var_dic.items():
        print('{:s}:{}'.format(name,value))