


class ClassDict:
    def __init__(self):
        self.int = 1
        self.double = 2.0
        self.str = 'str'

    def print_dict(self):
        dict = {}
        for attr, value in self.__dict__.items():
            if hasattr(value, "to_dict"):
                dict[attr] = value.to_dict()
            else:
                dict[attr] = value
            print(dict[attr])
        return dict

def print_class_dict():
    cd = ClassDict()
    cd.print_dict()


if __name__ == '__main__':
    print_class_dict()