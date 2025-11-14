import pandas as pd
import math
import numpy as np
from sklearn.cluster import DBSCAN

EPSILON = .2

def convert_to_day_seconds(ts: pd.Timestamp):
    ts = ts.round("1s")
    ts_hour = ts.hour
    ts_minute = ts.minute
    ts_second = ts.second
    day_seconds = ts_hour*3600 + ts_minute*60 + ts_second
    return day_seconds

def compute_frac_of_day(ts:pd.Timestamp):
    total_seconds_day = 24*3600
    return convert_to_day_seconds(ts)/total_seconds_day

def project_to_unit_circle(ts:pd.Timestamp):
    frac_of_day = compute_frac_of_day(ts)
    proj_unit_circle = (math.cos(2*math.pi*frac_of_day), math.sin(2*math.pi*frac_of_day))
    return proj_unit_circle

def compute_clustering(window_opening_times: list):
    # Compute the projection onto the unit circle for all window opening times
    projections_onto_circle = list(map(lambda x: project_to_unit_circle, window_opening_times))
    projections_onto_circle = np.array(projections_onto_circle)

    clustering = DBSCAN(eps=EPSILON, min_samples=2).fit(projections_onto_circle)
    clusters = {}
    for c in np.unique(clustering.labels_):
        ix = np.where(clustering.labels_ == c)
        clusters[c] = [window_opening_times[i] for i in ix]
    return clusters

def compute_clusters_boundaries(window_opening_times: list):
    clusters = compute_clustering(window_opening_times)
    clusters_boundaries = {}
    for c in clusters.keys():
        cluster_minimum = min(map(lambda x: x.time(), clusters[c]))
        cluster_maximum = max(map(lambda x: x.time(), clusters[c]))
        clusters_boundaries[c] = (cluster_minimum, cluster_maximum)

    return clusters_boundaries

