from bs4 import BeautifulSoup
import urllib.request
import re

PAGES = 382

with open('macdonell.tsv', 'w') as fout:
    for page in range(1, PAGES + 1):
        print(page)
        link = f"https://dsal.uchicago.edu/cgi-bin/app/macdonell_query.py?page={page}"
        with urllib.request.urlopen(link) as resp:
            soup = BeautifulSoup(resp, 'html.parser')
            for s in soup.find_all('div'):
                if s.find('hw', recursive=False):
                    word = s.hw.find_all('b')[1].text
                    fout.write(f'{word}\n')