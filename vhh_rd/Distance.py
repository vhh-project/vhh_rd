import yaml
from sklearn.neighbors import DistanceMetric

def get_dist(distance_type: str, metric_param: int):

    if distance_type in ["euclidean", "manhattan", "chebyshev"]:
        return DistanceMetric.get_metric(distance_type)
    elif distance_type == "minkowski":
        return DistanceMetric.get_metric(distance_type,  p=metric_param)
    else:
        raise ValueError("Unknonwn distance type")

class Distance(object):
    """
        A class that can be used to compute distances
    """
    def __init__(self, distance_type: str, metric_param: int):
        self.dist = get_dist(distance_type, metric_param)

    def __call__(self, x, y):
        return self.dist.pairwise([x, y])[0,1]
        

    