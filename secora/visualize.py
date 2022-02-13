from numpy.lib.npyio import NpzFile
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Visualize a highdimensional vector space')
  parser.add_argument('embedding', metavar='embedding', type=NpzFile, help='send npz file')
  args = parser.parse_args()

  embedding = args.embedding


#np.savez_compressed(embedding)
matrix = np.load(embedding)


standart_matrix = StandardScaler().fit_transform(matrix)
#scale data

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(standart_matrix)
principalDf = pd.DataFrame(data=principalComponents, columns= ['x','y'])


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.scatter(principalDf['x'], principalDf['y'])

variance1=(pca.explained_variance_ratio_[0])
variance2=(pca.explained_variance_ratio_[1])

print("The information in Demension 1 =", variance1*100,"%")
print("The information in Demension 2 =", variance2*100,"%")
print("Sum of information =", (variance1+variance2)*100, "%")
