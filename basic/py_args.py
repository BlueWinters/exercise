# coding=utf-8

def args_args_and_kwargs(*args, **kwargs):
    print(args)
    print(kwargs)

    # examples
    print(args[0]) # 可能报错
    print(args[1]) # 可能报错

    print(kwargs.get('size',10))
    print(kwargs.get('shape',[2,3]))

if __name__ == '__main__':
    # args_args_and_kwargs()
    # args_args_and_kwargs(1, 2)
    # args_args_and_kwargs(1, 2, size=1)
    # args_args_and_kwargs(size=1, shape=[3,4])
    args_args_and_kwargs(1, 2, size=1, shape=[3,4])