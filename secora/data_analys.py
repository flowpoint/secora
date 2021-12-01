import pickle
import pandas as pd
import json


#with open('/media/kai/Volume/datascience/CodeSearchNet/resources/data/java_dedupe_definitions_v2.pkl', 'rb') as f:
#with open('/media/kai/Volume/datascience/CodeSearchNet/resources/data/go_dedupe_definitions_v2.pkl', 'rb') as f:
#with open('/media/kai/Volume/datascience/CodeSearchNet/resources/data/python_dedupe_definitions_v2.pkl', 'rb') as f:
#with open('/media/kai/Volume/datascience/CodeSearchNet/resources/data/php_dedupe_definitions_v2.pkl', 'rb') as f:
#with open('/media/kai/Volume/datascience/CodeSearchNet/resources/data/javascript_dedupe_definitions_v2.pkl', 'rb') as f:
with open('/media/kai/Volume/datascience/CodeSearchNet/resources/data/ruby_dedupe_definitions_v2.pkl', 'rb') as f:
    data = pickle.load(f)

print("Total functions: ", len(data))
#print(type(data))
#print(data[1])



sum = 0
max_value = None
min_value = None

for i in range(len(data)):
    sum += (len(data[i]['docstring_tokens']))
    if (max_value is None or len(data[i]['docstring_tokens']) > max_value):
        max_value = len(data[i]['docstring_tokens'])
mean = sum/len(data)
print("Mean doc length: ", mean)
print("Max tokens: ", max_value)



sumc = 0
max_valuec = 0
min_valuec = 0
max_char = 0

for i in range(len(data)):
    sumc += (len(data[i]['function_tokens']))

    if (max_valuec is None or len(data[i]['function_tokens']) > max_valuec):
        max_valuec = len(data[i]['function_tokens'])

    if (min_valuec is None or len(data[i]['function_tokens']) < min_valuec):
        min_valuec = len(data[i]['function_tokens'])

    now_char = len(str(data[i]['function_tokens']))
    if (now_char > max_char):
        max_char = now_char

meanc = sumc/len(data)
print("Mean code length: ", meanc)
print("Max tokens: ", max_valuec)
print("Min: ", min_valuec)
#print(data[0]['function_tokens'])
print("Max char: ", max_char)


