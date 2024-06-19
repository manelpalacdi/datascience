def print_percentage(i: int, n: int):
    # print: |··········| {x} % trained
    percentage = float(i) / n * 100
    graphic = []
    for j in range(10):
        if (j * 10 < percentage):
            graphic.append('#')
        else:
            graphic.append('·')
    graphic = ''.join(graphic)
    if (i == 0):
        print(f"\n")
    elif (i == n - 1):
        print(f"\r\t------Training completed------\n\n")
    else:
        print(f"\r\t|{graphic}| {str(percentage):.2}% trained", end='')