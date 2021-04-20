# Import Packages
import numpy as np
from keras.datasets import fashion_mnist
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import keras
from keras import backend as K


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_testenc = np.expand_dims(X_test, axis=3)



model_name = 'autoencoder.h5'
loaded_model = keras.models.load_model(model_name)

encoder = K.function([loaded_model.layers[0].input], [loaded_model.layers[4].output])
encoded_images = encoder([X_testenc])[0].reshape(-1, 7*7*7)




X_test = X_test.reshape(len(X_test),-1)
X_test = X_test.astype(float) / 255.

feat_cols = [ 'pixel'+str(i) for i in range(X_test.shape[1]) ]
df = pd.DataFrame(X_test,columns=feat_cols)
df['y']=y_test
df['label'] = df['y'].apply(lambda i: str(i))
print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])
print('Size of the dataframe: {}'.format(df.shape))

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
plt.show()

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"],
    ys=df.loc[rndperm,:]["pca-two"],
    zs=df.loc[rndperm,:]["pca-three"],
    c=df.loc[rndperm,:]["y"],
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()


feat_cols = [ 'pixel'+str(i) for i in range(encoded_images.shape[1]) ]
df = pd.DataFrame(encoded_images,columns=feat_cols)
df['y']=y_test
df['label'] = df['y'].apply(lambda i: str(i))
print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])
print('Size of the dataframe: {}'.format(df.shape))

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
df['auto-one'] = encoded_images[:,0]
df['auto-two'] = encoded_images[:,1]
df['auto-three'] = encoded_images[:,2]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
plt.show()
plt.figure(figsize=(16,10))
sns.scatterplot(x="auto-one",
                y="auto-one",
                hue="y",
                palette=sns.color_palette("hls", 10),
                data=df.loc[rndperm, :],
                legend="full", alpha=0.3
                )
plt.show()
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["auto-one"],
    ys=df.loc[rndperm,:]["auto-two"],
    zs=df.loc[rndperm,:]["auto-three"],
    c=df.loc[rndperm,:]["y"],
    cmap='tab10'
)
ax.set_xlabel('auto-one')
ax.set_ylabel('auto-two')
ax.set_zlabel('auto-three')
plt.show()
# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=df.loc[rndperm,:]["pca-one"],
#     ys=df.loc[rndperm,:]["pca-two"],
#     zs=df.loc[rndperm,:]["pca-three"],
#     c=df.loc[rndperm,:]["y"],
#     cmap='tab10'
# )
# ax.set_xlabel('pca-one')
# ax.set_ylabel('pca-two')
# ax.set_zlabel('pca-three')
# plt.show()


