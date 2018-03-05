# -*- coding: utf-8 -*-


if __name__ == "__main__":
    m = input()
    n = input()
    x = input()
    # vasiot kod pisuvajte go tuka
    tablica = {}
    scope = range(int(m), int(n) + 1)
    for i in scope:
        result = i ** 3
        tablica[result] = i
    if int(x) not in tablica.keys():
        print("nema podatoci")
    else:
        print(tablica[int(x)])
    print(sorted(tablica.items()))
