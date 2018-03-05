def sovrshen_broj(number):
    # your code here
    sum = 0
    for i in range(1, number):
        if number % i == 0:
            sum += i
    return sum == number


if __name__ == "__main__":
    broj = int(input())
    isPerfect = sovrshen_broj(broj)
    # your code here
    if isPerfect:
        print("Brojot", broj, "e sovrshen")
    else:
        print("Brojot", broj, "ne e sovrshen")
