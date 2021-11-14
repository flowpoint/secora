import csv
import sys

csv.field_size_limit(sys.maxsize)

with open('predictions.csv', 'w') as csvfile:
    fieldnames = ['language', 'query', 'url']
    cwriter = csv.DictWriter(csvfile,fieldnames)
    cwriter.writeheader()

    for line in reader:
        q = 'convert int to string'
        l = 'python'
        u = 'https://github.com/raphaelm/python-sepaxml/blob/187b699b1673c862002b2bae7e1bd62fe8623aec/sepaxml/utils.py#L64-L76'

        cwriter.writerow(
            {'language': l,
                'query': q,
                'url': u
            })

