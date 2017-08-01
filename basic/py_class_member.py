# coding=utf-8


class Class(object):
    def __init__(self):
       self.counter = []

    def append(self, new):
        self.counter.append(new)

    def count(self):
        return self.counter

    def print(self):
        print(self.counter)

    def get_counter(self):
        counter = []
        for n in range(len(self.counter)):
            counter.append(self.counter[n])
        return counter


def class_member_set():
    c = Class()

    c.append(1)
    c.print()
    c.append(2)
    c.print()

    counter1 = c.count()
    counter1.extend([3])
    print(counter1)
    c.print()

    counter2 = c.get_counter()
    counter2.extend([4,5])
    print(counter2)


    # 输出
    # [1]
    # [1, 2]
    # [1, 2, 3]
    # [1, 2, 3]
    # [1, 2, 3, 4, 5]



if __name__ == '__main__':
    class_member_set()