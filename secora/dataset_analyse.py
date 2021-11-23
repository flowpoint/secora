from pathlib import Path
import pandas as pd

python_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/python').glob('**/*.gz'))
java_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/java/').glob('**/*.gz'))
go_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/go/').glob('**/*.gz'))
php_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/php/').glob('**/*.gz'))
javascript_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/javascript/').glob('**/*.gz'))
ruby_files = sorted(Path('/media/kai/Volume/datascience/CodeSearchNet/resources/data/ruby/').glob('**/*.gz'))
all_files = python_files + go_files + java_files + php_files + javascript_files + ruby_files

print(f'Total number of files: {len(all_files):,}')




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
df = jsonl_list_to_dataframe(php_files, columns_short_list)



code_list = df['code_tokens']
doc_list = df['docstring_tokens']

print("Code length:")
print("Min: ", len(min(code_list, key=len)))
print("Max: ", len(max(code_list, key=len)))

print("Documentation length:")
print("Min: ", len(min(doc_list, key=len)))
print("Max: ", len(max(doc_list, key=len)))

print(len(min(doc_list, key=len)))

print("The number of functions is: ", len(code_list))



def average_words(list):
    count = 0
    for listElem in list:
        count += len(listElem) 

    return count/len(list)                   

print("The average code words per function are: ", average_words(code_list))
print("The average documentation words per function are: ", average_words(doc_list))




 
