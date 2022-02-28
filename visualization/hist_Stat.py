from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Visualize the length of the codesearchnet dataset in a histogramm 


#the whole dataset might be to big for ram, also the analyse of the dataset seperated into the different languages can be necessary 
#uncomment the files to analyse
python_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/python').glob('**/*.gz'))
#java_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/java/').glob('**/*.gz'))
#go_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/go/').glob('**/*.gz'))
#php_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/php/').glob('**/*.gz'))
#javascript_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/javascript/').glob('**/*.gz'))
#ruby_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/ruby/').glob('**/*.gz'))
#all_files = python_files + go_files + java_files + php_files + javascript_files + ruby_files

#print(f'Total number of files: {len(all_files):,}')




columns_long_list = ['repo', 'path', 'url', 'code', 
                     'code_tokens', 'docstring', 'docstring_tokens', 
                     'language', 'partition']

columns_short_list = ['code_tokens', 'docstring_tokens', 
                      'language', 'partition']

def jsonl_list_to_dataframe(file_list, columns=columns_short_list):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    return pd.concat([pd.read_json(f, 
                                   orient='records', 
                                   compression='gzip',
                                   lines=True)[columns] 
                      for f in file_list], sort=False)

#insert the language to analys, like "...datafram(java_files,..."
df = jsonl_list_to_dataframe(python_files, columns_short_list)


#comment out to chose code or documentation
list = df['code_tokens']
#list = df['docstring_tokens']

#counting words in in the dataframe
count = []
for listElem in list:
        count.append(len(listElem)) 
#print(count)

#outlier filter to expand the inlier bigger scaled
def outlier_filter(count):
    l = np.array((count), int)
    l = np.array(l)
    l = l[(l>np.quantile(l,0.1)) & (l<np.quantile(l,0.9))].tolist()
    return l

#manually sum of words in bars for code and doc over all languages
doc = [1033963, 610145, 188213, 87231, 46189, 27106, 17468, 12808, 7823, 39590]
code = [0, 68, 142574, 237991, 253801, 201443, 168591, 135363, 113099, 817606]
bins=range(0, 101, 10)

#helper function to creaate list for histogramm plot
#values=[]
#for i in range(len(doc)):
#    for j in range(doc[i]):
#        values.append((i+1)*10-1)




fig, ax = plt.subplots()
squad = ['0','10','20','30', '40', '50', '60', '70', '80', '90', 'inf']#hardcoding bars
ax.set_xticks(bins)
ax.set_xticklabels(squad, minor=False)

plt.hist(np.clip(count, bins[0], bins[-1]), bins=bins)
plt.title("Python Codelänge Verteilung")
plt.xlabel("Anzahl Wörter")
plt.ylabel("Häufigkeit")
plt.show()


