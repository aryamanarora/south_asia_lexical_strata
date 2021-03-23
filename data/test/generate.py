import random
import os
import csv

vowels = ['a', 'e', 'i', 'o', 'u']
consonants = ['m', 'n', 'p', 'b', 't', 'd', 'k', 'g', 's', 'z', 'r', 'l', 'h', 'v', 'w', 'sh']

A = 800
B = 200

ress = []
for x in range(10):
    print(x)
    with open('out.txt', 'w') as fout:
        for _ in range(A):
            count = random.randrange(1, 6)
            word = []
            for i in range(count):
                syllable = [random.choice(consonants), random.choice(vowels)]
                word += syllable
            fout.write(' '.join(word) + '\n')
        for _ in range(B):
            count = random.randrange(1, 6)
            word = []
            for i in range(count):
                syllable = [random.choice(consonants), random.choice(vowels), random.choice(consonants)]
                word += syllable
            fout.write(' '.join(word) + '\n')

    os.system("python ../../models/main_multiple_rnns.py out.txt out.txt result.txt")

    res = {(1, 1): 0, (1, 2): 0, (2, 1): 0, (2, 2): 0}
    with open('result.txt', 'r') as fin:
        reader = csv.reader(fin, delimiter='\t')
        cur = 1
        for i, row in enumerate(reader):
            if i >= A:
                cur = 2
            if float(row[2]) > float(row[1]):
                res[(cur, 1)] += 1
            else:
                res[(cur, 2)] += 1
    ress.append(res)
print(ress)