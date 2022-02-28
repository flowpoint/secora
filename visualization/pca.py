import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.savez_compressed('data.npz.npy')
matrix = np.load('data.npz.npy')


standart_matrix = StandardScaler().fit_transform(matrix)
#scale data

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(matrix)
principalDf = pd.DataFrame(data=principalComponents, columns= ['x','y'])


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.scatter(principalDf['x'], principalDf['y'])

variance1=(pca.explained_variance_ratio_[0])
variance2=(pca.explained_variance_ratio_[1])

print("The information in Demension 1 =", variance1*100,"%")
print("The information in Demension 2 =", variance2*100,"%")
print("Sum of information =", (variance1+variance2)*100, "%")