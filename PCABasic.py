from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

x, y = make_blobs(n_samples=100, n_features=10)

print("Before =",x.shape)

pca = PCA(n_components=4)

pca.fit(x)
x = pca.transform(x)

print("After =",x.shape)


