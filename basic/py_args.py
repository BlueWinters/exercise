# coding=utf-8
import argparse

def args_args_and_kwargs(*args, **kwargs):
    print(args)
    print(kwargs)

    # examples
    print(args[0]) # 可能报错
    print(args[1]) # 可能报错

    print(kwargs.get('size',10))
    print(kwargs.get('shape',[2,3]))

def for_args_command():
    def to_dict(args):
        # 转换成一个dict
        dc = vars(args)
        print(dc)
        return dc

    def print(args):
        # 直接以str的形式输出
        print(str(args))

    parser = argparse.ArgumentParser(description="command line description")
    parser.add_argument('--model_name', type=str, default='vggnet', help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    args = parser.parse_args() # 分析命令，转换为一个namespace

    to_dict(args)
    print(args)



if __name__ == '__main__':
    # args_args_and_kwargs()
    # args_args_and_kwargs(1, 2)
    # args_args_and_kwargs(1, 2, size=1)
    # args_args_and_kwargs(size=1, shape=[3,4])
    # args_args_and_kwargs(1, 2, size=1, shape=[3,4])

    for_args_command()