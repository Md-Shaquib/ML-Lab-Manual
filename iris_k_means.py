import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
print(iris.feature_names)
print(iris)
#print(iris.describe())

df = pd.DataFrame({
    'X': iris.data[:,0],
    'Y': iris.data[:,1],
    'cluster':iris.target
})
print(df)

centroids = {}
for i in range(3):
    result_list = []
    result_list.append(df.loc[df['cluster']==i]['X'].mean())
    result_list.append(df.loc[df['cluster']==i]['Y'].mean())

    centroids[i] = result_list

print(centroids)

fig = plt.figure(figsize=(5,5))
plt.scatter(df['X'],df['Y'], c=iris.target, cmap='gist_rainbow')
plt.xlabel('Sepal Length', fontsize = 18)
plt.ylabel('Sepal Width', fontsize =18)
plt.show()

colmap = {0: 'r', 1: 'g',2:'b'}
for i in range(3):
    plt.scatter(centroids[i][0],centroids[i][1], color =colmap[i])
plt.show()

fig = plt.figure(figsize=(5,5))
plt.scatter(df['X'], df['Y'], c=iris.target, alpha =0.3)
colmap={0: 'r', 1 :'g' , 2: 'b'}
col = [0,1]
for i in centroids.keys():
    plt.scatter(centroids[i][0],centroids[i][1], c=colmap[i], edgecolor='k')
plt.show()


def calculation (df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)]=(
            np.sqrt(
                (df['X']- centroids[i][0])**2
                + (df['Y']-centroids[i][1])**2
            )
        )
    centroid_distance_cols = ['distance_from_{}' .format(i) for i in centroids.keys()]
    df['closest']= df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest']=df['closest'].map(lambda X: int(X.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda X: colmap[X])
    return df

df = calculation(df,centroids)
print(df)

fig = plt.figure(figsize=(5,5))
plt.scatter(df['X'], df['Y'], color=df['color'], alpha =0.3)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], edgecolor='k')
plt.show()

def update(k):
    for i in range(3):
        centroids[i][0]=np.mean(df[df['closest']==i]['X'])
        centroids[i][1]=np.mean(df[df['closest']==i]['Y'])
    return k

centroids = update(centroids)
print(centroids)

fig = plt.figure(figsize=(5,5))
ax=plt.axes()
plt.scatter(df['X'], df['Y'], color=df['color'], alpha =0.3)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], edgecolor='k')
plt.show()

df=calculation(df, centroids)

fig = plt.figure(figsize=(5,5))
plt.scatter(df['X'], df['Y'], color=df['color'], alpha =0.3)
for i in centroids.keys():
    plt.scatter(centroids[i][0],centroids[i][1], color=colmap[i], edgecolor='k')
plt.show()

while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df= calculation(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

fig = plt.figure(figsize=(5,5))
plt.scatter(df['X'], df['Y'], color=df['color'])
for i in centroids.keys():
    plt.scatter(centroids[i][0],centroids[i][1], color=colmap[i], edgecolor='k')
plt.show()
