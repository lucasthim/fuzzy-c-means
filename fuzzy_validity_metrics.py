import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.linalg import norm


class FuzzyClusteringValidatityMetrics:
    
      """
      Some validity metrics for Fuzzy C-Means:

      Partition Coefficient: Measures 'fuzziness' in partitioned clustering.
      
      Partition Entropy: Measures entropy in partitioned clustering.

      """

      def __init__(self):
        self.__fpc = None
        self.__fpe = None
        self.__gSil_items = None
        self.__gSil = None

        self.u_membership = None

      def fpc(self, u_membership: np.array = None):
            """
            Fuzzy partition coefficient relative to fuzzy c-partitioned
            membership matrix. Measures 'fuzziness' in partitioned clustering.
            
            Parameters:

            u_membership (2D numpy.array): Fuzzy membership matrix (n x c) where
            n = number of data points and c = number of clusters.

            Returns:

            fpc (float): Fuzzy partition coefficient.
            """
            if(self.__fpc is not None and u_membership is None):
              return self.__fpc
            n_data_points = np.float(u_membership.shape[0])
            self.__fpc = (u_membership ** 2).sum() / n_data_points
            return self.__fpc

      def fpe(self, u_membership: np.array = None):
            """
            Fuzzy partition entropy relative to fuzzy c-partitioned
            membership matrix.
            
            Parameters:

            u_membership (2D numpy.array): Fuzzy membership matrix (n x c) where
            n = number of data points and c = number of clusters.

            Returns:

            fpe (float): Fuzzy partition entropy.
            """
            if(self.__fpe is not None and u_membership is None):
              return self.__fpe
            n_data_points = np.float(u_membership.shape[0])
            self.__fpe = (u_membership * np.log(u_membership)).sum()/(-n_data_points)
            return self.__fpe

      def __calculate_parwise_minimum(self, u_membership, sample, reference_pair,n_samples):
          pairwise_minimum =  np.tile(np.expand_dims(sample, axis=1),n_samples).T
          pairwise_minimum = np.minimum(pairwise_minimum,u_membership)
          pairwise_minimum[reference_pair,:] = 0
          return pairwise_minimum

      def __calculate_minimum_score_weighted_mean(self, parwise_minimum,pairwise_distance):
          numerator = (parwise_minimum * pairwise_distance).sum(axis=0)
          denominator = parwise_minimum.sum(axis=0)
          denominator[denominator == 0] = np.nan
          return numerator / denominator

      def generalized_silhouette(self, u_membership = None,X: np.array = None):
          
          """
          Generalized Silhouette measure for fuzzy partitions.
          (Rawashdeh & Ralescu) 2012.
          Parameters:

          X (2D array): matrix of datapoints (n x k) where
          n = number of data points and k = number of dimensions.

          u_membership (2D numpy.array): Fuzzy membership matrix (n x c) where
          n = number of data points and c = number of clusters.

          Returns:

          gSil (float): generalized silhouette coefficient.
          """    
          if(self.__gSil is not None and u_membership is None and X is None):
              return self.__gSil
          if(self.__gSil is None and (u_membership is None or X is None)):
            return -2

          if(type(u_membership) == pd.DataFrame):
            u_membership = u_membership.values

          a = []
          b = []          
          n_clusters = u_membership.shape[1]
          n_samples = u_membership.shape[0]

          pairwise_distance = distance.cdist(X,X)
          u_membership_shifted = np.roll(u_membership,-1,axis=1)

          for index,sample in enumerate(u_membership):
              inter_dist_1 =  self.__calculate_parwise_minimum(u_membership_shifted,sample,index,n_samples)
              inter_dist_2 =  self.__calculate_parwise_minimum(u_membership,np.roll(sample,-1),index,n_samples)
              inter_distance = np.maximum(inter_dist_1,inter_dist_2)

              intra_distance = self.__calculate_parwise_minimum(u_membership,sample,index,n_samples)

              dist_k = pairwise_distance[:,index]
              dist_k = np.tile(np.expand_dims(dist_k, axis=1),n_clusters)
              
              a.append(self.__calculate_minimum_score_weighted_mean(intra_distance,dist_k))
              b.append(self.__calculate_minimum_score_weighted_mean(inter_distance,dist_k))

          a = np.nanmin(a,axis=1)
          b = np.nanmin(b,axis=1)
          self.__gSil_items = (b - a) / np.maximum(b,a)
          self.__gSil = self.__gSil_items.mean()
          return self.__gSil

      def all(self, u_membership: np.array,X):
          self.fpc(u_membership)
          self.fpe(u_membership)
          self.generalized_silhouette(u_membership,X)
          