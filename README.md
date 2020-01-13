# Fuzzy C Means

This project presents an implementation of the Fuzzy C-Means clustering algorithm, along with some validity metrics for fuzzy clusterings.

Fuzzy Clustering Validity Measures available:

- Generalized Silhouette
- Fuzzy Partition Coefficient
- Fuzzy Entropy

## Installation

Download this repo and import it to the desired directory.
Package in PyPi will be available soon.

## Usage

```python
from fuzzy_c_means import FuzzyCMeans
from fuzzy_validity_metrics import FuzzyClusteringValidatityMetrics
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter
import time

# create artifitial dataset
n_samples = 3000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here
centers = [(-5, 0), (0, 0), (10, 2)]

X,_ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=2,
                  centers=centers, shuffle=False, random_state=42)

def plot_data_with_centroids(X,fcm_centers,fcm_labels):
  plt.style.use('seaborn-whitegrid')
  f, axes = plt.subplots(1, 1, figsize=(10,10))
  scatter(X[:,0], X[:,1], ax=axes, hue=fcm_labels)
  scatter(fcm_centers[:,0], fcm_centers[:,1], ax=axes,marker="s",s=200)
  plt.grid(True)
  plt.title('FCM clustering result')
  plt.axis('equal')
  plt.show()
  
start_time = time.time()

fcm = FuzzyCMeans(m=1.5)
fcm.fit(dataset=X,n_clusters=3,tolerance=0.001,verbose=1,max_iterations=100)
fcm_centers = fcm.centroids
fcm_labels  = fcm.u_membership.argmax(axis=1)

print('')
plot_data_with_centroids(X,fcm_centers,fcm_labels)
d_time = (time.time() - start_time)
print('')
print("--- Total run time: %.3f seconds ---" % d_time)
print('')
print('Final centroids: ')
print(fcm_centers)  

```

## Contributing
Pull requests are more than welcome! For major changes, please open an issue first to discuss what you would like to change.


## Future Work
An Optimization of Fuzzy Clustering is under development and can be seen in the file Hybrid_FCM_PSO_Clustering. Optimization process is done using the Particle Swarm Optimization algorithm to mitigate the chance of a bad clustering initialization.
## License
[MIT](https://choosealicense.com/licenses/mit/)
