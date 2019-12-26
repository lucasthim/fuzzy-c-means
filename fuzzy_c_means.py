import numpy as np
from scipy.spatial import distance
from scipy.linalg import norm
from fuzzy_validity_metrics import FuzzyClusteringValidatityMetrics

class FuzzyCMeans:
    
      """
      Fuzzy C-Means implementation based on  J. C. Bezdek, 
      Pattern Recognition with Fuzzy Objective Function Algorithms (1981).
      <https://doi.org/10.1007/978-1-4757-0450-1>

      """
      def __init__(self,random_state = None,m = 2,n_clusters = 10):

            self.random_state = random_state
            if(random_state is not None):
              random = np.random.RandomState(random_state)
            else:
              random = np.random.RandomState()
            self.random = random
            self.m = m
            self.n_clusters = n_clusters
            self.n_dim = None
            self.u_initial, self.centroids_initial = None, None
            self.u_membership, self.centroids = None, None
            self.validity_metrics, self.n_data_points  =None, None

      def initialize_u_membership(self):

            """
            Initialize membership matrix U where the sum of lines is equal to 1.

            Parameters:
             ------x------

            Returns:

            u_membership (np.array): fuzzy membership matrix (n x c) where
            n = number of data points and c = number of clusters.

            """
            random_matrix = self.random.rand(self.n_data_points,self.n_clusters)
            return random_matrix / np.tile(np.reshape(random_matrix.sum(axis=1),(self.n_data_points,1)),self.n_clusters)
            
      def calculate_cluster_centroids(self, u_membership_input: np.array,data_points: np.array):
            """
            Calculate centroids for each cluster based on data points and current 
            membership matrtix.

            Parameters:

            data_points (2D numpy.array): matrix of datapoints (n x k) where
            n = number of data points and k = number of dimensions.

            u_membership_input (2D numpy.array): fuzzy membership matrix (n x c) where
            n = number of data points and c = number of clusters.

            m (float): fuzzy coefficient

            Returns:

            centroids (np.array): calculated centroids for c number of clusters.

            """

            u_membership = u_membership_input**self.m
            numerator = np.dot(u_membership.T,data_points)

            denominator = np.reshape((u_membership).sum(axis=0), (self.n_clusters,1))
            denominator = np.tile(denominator,self.n_dim)

            self.centroids =  numerator / denominator
            return self.centroids

      def calculate_membership_matrix(self, data_points: np.array, centroids:np.array):
            
            """
            Calculate membership matrix based on current centroids, data points 
            and fuzzy coefficient.

            Parameters:

            data_points (2D numpy.array): matrix of datapoints (n x k) where
            n = number of data points and k = number of dimensions.

            centroids (np.array): calculated centroids for c number of clusters.

            """
            if (np.float(self.m) <= 1.02):
              print(self.m)
              self.m = 1.02

            distance_to_centroids_matrix = distance.cdist(data_points,centroids)

            inv_dist = distance_to_centroids_matrix**(-1)
            numerator = inv_dist ** (2/(self.m - 1))

            denominator = (inv_dist ** (2/(self.m - 1))).sum(axis=1) 
            denominator = np.tile(np.reshape(denominator,(self.n_data_points,1)),self.n_clusters)
            u_membership = numerator / denominator
            return u_membership


      def fit(self, dataset: np.array, n_clusters: int, max_iterations = 1000,tolerance = 0.000001,verbose = 0):
          
            """
            Execute the FCM algorithm.

            Parameters:

            data (2D numpy.array): dataset (n x k) where
            n = number of data points and k = number of dimensions.
            
            n_clusters (int): number of clusters to partition the dataset

            max_iterations (int): maximum number of iterations.

            tolerance (float): minimum tolerance to stop iterations.

            u_membership (2D numpy.array): fuzzy membership matrix (n x c) where
            
            m (float): fuzzy coefficient
            """
            self.n_clusters = n_clusters
            self.n_data_points = dataset.shape[0]
            self.n_dim = dataset.shape[1]

            self.u_initial = self.initialize_u_membership()
            self.centroids_initial = self.calculate_cluster_centroids(
                                      u_membership_input = self.u_initial,
                                      data_points = dataset)

            u_current_state = self.u_initial.copy()
            for iteration in range(1,max_iterations):    
                
                if(verbose >= 1):
                  print('Iteration: %.0f ' % iteration)

                centroids = self.calculate_cluster_centroids(u_membership_input = u_current_state,data_points = dataset)
                u_next_state = self.calculate_membership_matrix(data_points = dataset,centroids = centroids)
                
                current_tolerance = norm(u_next_state - u_current_state)
                u_current_state = u_next_state

                if (current_tolerance < tolerance):
                  break
            self.u_membership = u_current_state
            self.centroids = centroids
            self.validity_metrics = FuzzyClusteringValidatityMetrics()
            self.validity_metrics.all(self.u_membership,dataset)
            if(verbose >= 1):
              print('')
              print('Centroids : ')
              print(centroids)
              print('')
              print('Validity Measures : ')
              print('Fuzzy Partition Coefficient: %.4f' % self.validity_metrics.fpc())
              print('Fuzzy Partition Entropy: %.4f' % self.validity_metrics.fpe())
              print('Generalized Silhouette: %.4f' % self.validity_metrics.generalized_silhouette())

