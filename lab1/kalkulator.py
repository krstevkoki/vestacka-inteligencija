# -*- coding: utf-8 -*-
__operators = ('+', '-', '/', '//', '*', '**', '%')


def calculator():
    x = input()
    operator = input()
    y = input()

    # your code here
    rezultat = 0
    if operator not in __operators:
        print("Operator", operator, "is not valid!")
    else:
        rezultat = eval(str(x) + operator + str(y))
        print(rezultat)
    return rezultat


if __name__ == "__main__":
    calculator()
