# -*- coding: utf-8 -*-
from math import sqrt

if __name__ == "__main__":
    m = input()
    n = input()
    x = input()
    # vasiot kod pisuvajte go tuka
    scope = range(int(m), int(n) + 1)
    tablica = {}
    for i in scope:
        tablica[i] = (i ** 2, i ** 3, round(sqrt(i), 5))
    if int(x) not in scope:
        print("nema podatoci")
    else:
        print(tablica[int(x)])
    print(sorted(tablica.items()))
