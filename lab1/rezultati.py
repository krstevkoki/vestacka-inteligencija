def suma_kolokviumi(rezultati):
    # your code here
    for r in rezultati:
        r["Vkupno od kolokviumi"] = r["Kolokvium 1"] + r["Kolokvium 2"]
        del r["Kolokvium 1"]
        del r["Kolokvium 2"]
    return rezultati


if __name__ == "__main__":
    n = int(input())
    rezultati = []  # ova e listata od rechnici
    for i in range(0, n):
        r = {}  # rechnik koj kje chuva podatoci za eden student
        brojIndeks = int(input())
        brojPoeni1 = int(input())
        brojPoeni2 = int(input())
        # ovde dodadete gi podatocite vo rechnikot. Potoa dodadete go rechnikot vo listata rezultati!!
        r["indeks"] = brojIndeks
        r["Predmet"] = "Veshtachka inteligencija"
        r["Kolokvium 1"] = brojPoeni1
        r["Kolokvium 2"] = brojPoeni2
        rezultati.append(r)
    print(suma_kolokviumi(rezultati))
