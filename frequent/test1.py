
def out_func(x: int):
    def inner_func(y: int) -> int:
        return x + y
    return inner_func

if __name__ == '__main__':
    remain = 19
    remain -= 10
    print(remain)