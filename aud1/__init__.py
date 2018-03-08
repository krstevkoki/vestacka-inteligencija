""" Python OOP vezbi"""


class Student(object):
    def __init__(self, full_name="", age=0):
        self.full_name = full_name
        self.age = age

    def __repr__(self):
        return "Full Name: {name}\nAge: {age}".format(name=self.full_name, age=self.age)

    def __eq__(self, other):
        return self.age == other.age

    def get_age(self):
        return self.age


class Sample(object):
    x = 23

    def __init__(self):
        self.local_x = 23

    def increment(self):
        self.__class__.x += 1
        self.local_x += 2

    def __repr__(self):
        return "{x} - {local_x}".format(x=self.__class__.x, local_x=self.local_x)


class Parent(object):
    parentAttr = 100

    def __init__(self):
        print("Calling parent constructor")

    def parent_method(self):
        print("Calling parent method")

    def set_attr(self, attr):
        Parent.parentAttr = attr

    def get_attr(self):
        print("Parent attribute:", self.__class__.parentAttr)


class Child( Parent):
    def __init__(self):
        print("Calling child constructor")

    def child_method(self):
        print("Calling child method")

    def get_attr(self):
        print("Child attribute", Parent.parentAttr)


if __name__ == "__main__":
    s1 = Student("Kostadin Krstev", 19)
    s2 = Student("Kristina Andonovska", 20)
    print(s1)
    print(s2)
    print(hasattr(s1, "__eq__"))
    print(hasattr(s2, "__eq__"))
    print(s1 == s2)

    sample = Sample()
    print(sample)
    sample.increment()
    print(sample)
    sample.increment()
    print(sample.__class__.x, end=" ")
    print("-", end=" ")
    print(sample.local_x)

    c = Child()
    c.child_method()
    c.parent_method()
    c.set_attr(200)
    c.get_attr()
