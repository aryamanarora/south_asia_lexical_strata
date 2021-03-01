from bs4 import BeautifulSoup
import urllib.request
import re

PAGES = 806
CODES = {
    'Pk': 'Prakrit', 'K': 'Kannada', 'Ap': 'Apabhramsha', 'Ar': 'Arabic', 'Deś': 'Deshiya', 'Grk': 'Greek',
    'Guj': 'Gujarati', 'Koṅk': 'Konkani', 'Pā': 'Pali', 'Pers': 'Persian', 'Rājas': 'Rajasthani', 'Tam': 'Tamil',
    'Tel': 'Telugu', 'Ved': 'Vedic Sanskrit'
}

with open('tulpule_old_marathi.tsv', 'w') as fout:
    for page in range(1, PAGES + 1):
        print(page)
        link = f"https://dsal.uchicago.edu/cgi-bin/app/tulpule_query.py?page={page}"
        with urllib.request.urlopen(link) as resp:
            soup = BeautifulSoup(resp, 'html.parser')
            for s in soup.find_all('div'):
                if s.find('hw', recursive=False):
                    word = s.hw.find_all('b')[1].text
                    etyma = []
                    text = re.search(r'\[.*?\]', s.text)

                    if text:
                        text = text.group(0)
                        if 'Sk.' in text:
                            if 'Sk.]' in text: etyma.append('Sanskrit')
                            else: etyma.append('Sanskrit-derived')
                        for lang in CODES:
                            if lang + '.' in text: etyma.append(CODES[lang])

                    fout.write(f'{word}\t{",".join(etyma)}\n')