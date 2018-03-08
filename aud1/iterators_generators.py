class FibIterator(object):
    def __init__(self):
        self.fn2 = 1
        self.fn1 = 1

    def __iter__(self):
        return self

    def __next__(self):
        (self.fn1, self.fn2, old) = (self.fn1 + self.fn1, self.fn1, self.fn2)
        if old > 100:
            raise StopIteration
        return old


def generator():
    x = 2
    y = 3
    yield x, y, x + y
    z = 12
    yield z // x
    yield z // y
    return


def getword(file):
    for line in file:
        for word in line.split():
            yield word
    return


if __name__ == "__main__":
    f = open("/home/krstevkoki/Desktop/file.txt")
    g = getword(f)
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(g))

    fib = FibIterator()
    l = list(fib)
    print(l)
    print(sum(l))
    for i in fib:
        print(i)

    g = generator()
    print(next(g))
    print(next(g))
    print(next(g))
    # print(next(g))
