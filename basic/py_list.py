# coding=utf-8

def diff_list_extend_append():
    # 返回一个列表
    def new_list():
        return [1, 2]

    #
    ex_list1 = [5, 6]
    ex_list2 = (7, 8)

    ap_list1 = [5, 6]
    ap_list2 = (7, 8)
    ap_list3 = '9'
    ap_list4 = {'age':10}

    # extend方法
    l = new_list()
    l.extend(ex_list1)
    print(l)

    l = new_list()
    l.extend(ex_list2)
    print(l)

    # append方法
    l = new_list()
    l.append(ap_list1)
    print(l)

    l = new_list()
    l.append(ap_list2)
    print(l)

    l = new_list()
    l.append(ap_list3)
    print(l)

    l = new_list()
    l.append(ap_list4)
    print(l)

    # 输出
    # [1, 2, 5, 6]
    # [1, 2, 7, 8]
    # [1, 2, [5, 6]]
    # [1, 2, (7, 8)]
    # [1, 2, '9']
    # [1, 2, {'age': 10}]



if __name__ == '__main__':
    diff_list_extend_append()