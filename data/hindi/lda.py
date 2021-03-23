from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import csv

data = []
words = []
with open('mcgregor.csv', 'r') as fin:
    reader = csv.reader(fin)
    for ct, row in enumerate(reader):
        if ct % 1000 == 0: print(ct)
        words.append(row[0])
        word = []
        letters = list(row[0].split())
        word.extend(letters)
        letters = ['#'] + letters + ['#']
        for cut in range(2, 3):
            for i in range(len(letters) - cut + 1):
                word.append(' '.join(letters[i:i + cut]))
        data.append(word)

common_dictionary = Dictionary(data)
common_corpus = [common_dictionary.doc2bow(text) for text in data]
print(common_corpus[:5])

lda = LdaModel(common_corpus, num_topics=3)

with open('results/lda_3.csv', 'w') as fout:
    for ct, i in enumerate(common_corpus):
        if ct % 1000 == 0: print(ct)
        fout.write(f'{words[ct]}')
        for j in lda[i]:
            fout.write(f'\t{j[1]}')
        fout.write('\n')


