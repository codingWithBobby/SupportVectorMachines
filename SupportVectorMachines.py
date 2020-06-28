import numpy as np 
from pylab import *
from sklearn.preprocessing import MinMaxScaler 
from sklearn import svm, datasets


#creating fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
	np.random.seed(1234)  
	pointsPerCluster = float(N)/k
	X = []
	y = [] 
	for i in range(k):  
		incomeCentroid = np.random.uniform(2000.0, 200000) 
		ageCentroid = np.random.uniform(20.0, 70.0) 
		for j in range(int(pointsPerCluster)): 
			X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)]) 																										
			y.append(i) 

	X = np.array(X)  
	y = np.array(y)
	return X, y

#preparing our data

(X, y) = createClusteredData(100, 5) 


plt.figure(figsize=(8, 6)) 
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float)) 
plt.show()

scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X) 
X = scaling.transform(X) 


plt.figure(figsize=(8, 6)) 
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float)) 
plt.show()

#Applying SVC

C = 1.0 
svc = svm.SVC(kernel='sigmoid', C=C).fit(X, y) 


def plotPredictions(clf):
	#creating a dense grid of points to sample
	xx, yy = np.meshgrid(np.arange(-1, 1, .001),
				np.arange(-1, 1, .001))

	#converting it to numpy arrays
	npx = xx.ravel()
	npy = yy.ravel()

	#convert to a list of 2d points, which will hold our incomes and age
	samplePoints = np.c_[npx, npy]

	##generating predicited labels
	Z = clf.predict(samplePoints)

	plt.figure(figsize=(8,6))
	Z = Z.reshape(xx.shape)#reshaping results to match xx dimensions
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8) 
	plt.scatter(X[:,0], X[:,1], c=y.astype(np.float)) 
	plt.show()

plotPredictions(svc)

