import csv

res = {(1, 1): 0, (1, 2): 0, (2, 1): 0, (2, 2): 0}
with open('result.txt', 'r') as fin:
    reader = csv.reader(fin, delimiter='\t')
    cur = 1
    for i, row in enumerate(reader):
        if i >= 900:
            cur = 2
        if float(row[2]) > float(row[1]):
            res[(cur, 1)] += 1
        else:
            res[(cur, 2)] += 1
print(res)