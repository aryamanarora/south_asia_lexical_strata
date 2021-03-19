from bs4 import BeautifulSoup
import urllib.request
import re

PAGES = 1082

with open('mcgregor_etym.csv', 'w') as fout:
    for page in range(1, PAGES + 1):
        print(page)
        link = f"https://dsal.uchicago.edu/cgi-bin/app/mcgregor_query.py?page={page}"
        with urllib.request.urlopen(link) as resp:
            soup = BeautifulSoup(resp, 'html.parser')
            for s in soup.find_all('div'):
                if s.find('hw', recursive=False):
                    word = s.hw.b.text
                    etyma = set()
                    text = re.search(r'\[.*?\]', s.text)

                    if text:
                        text = text.group(0)
                        if 'S.' in text or 'Sk.' in text:
                            etyma.add('Sanskrit')
                        if ('H.' in text) or (not '.' in text) or ('cf.' in text):
                            etyma.add('Native')
                        if 'ad.' in text:
                            etyma.add('Semi-native')
                        if ('A.' in text) or ('P.' in text):
                            etyma.add('Perso-Arabic')
                        if 'Ap.' in text or 'Pa.' in text or 'Pk.' in text:
                            etyma.add('Native')
                        if 'Austro-as.' in text:
                            etyma.add('Munda')
                        if 'Drav.' in text:
                            etyma.add('Dravidian')
                        if 'Engl.' in text:
                            etyma.add('English')
                        if 'Pt.' in text:
                            etyma.add('Portuguese')

                    fout.write(f'{word}\t{",".join(list(etyma))}\n')