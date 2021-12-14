import csv
import sys

raise NotImplementedError()

csv.field_size_limit(sys.maxsize)

with open('resources/queries.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    queries = []
    for row in reader:
        queries.append(row)

with open('predictions.csv', 'w') as csvfile:

    fieldnames = ['language', 'query', 'url']
    cwriter = csv.DictWriter(csvfile,fieldnames)
    cwriter.writeheader()
    top_k = 10

    for query in queries:
        prediction = model.search(q, top_k)
        for p in prediction:
            language = p['language']
            url = p['language']

            # prediction is a list of samples
            predictions.append(prediction)

            cwriter.writerow(
                {'language': language,
                    'query': query,
                    'url': url
                })

