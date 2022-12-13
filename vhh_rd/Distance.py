import yaml
from sklearn.neighbors import DistanceMetric
import sklearn
import numpy as np

def get_dist(distance_type: str, metric_param: int):
    if distance_type in ["euclidean", "manhattan", "chebyshev"]:
        return DistanceMetric.get_metric(distance_type)
    elif distance_type == "minkowski":
        return DistanceMetric.get_metric(distance_type,  p=metric_param)
    return None
    

class Distance(object):
    """
        A class that can be used to compute distances
    """
    def __init__(self, distance_type: str, metric_param: int):
        self.distance_type = distance_type
        self.dist = get_dist(distance_type, metric_param)

    def __call__(self, x, y):
        if self.distance_type == "cosine":
            return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

        return self.dist.pairwise([x, y])[0,1]
        

    