import pandas as pd
import numpy as np
from .create_clustering import compute_frac_of_day

def compute_second_momentum(list_of_ts:list):
    list_of_frac = list(map(compute_frac_of_day, list_of_ts))
    mean = np.mean(list_of_frac)
    non_avg_momentum = 0
    for frac in list_of_frac:
        non_avg_momentum = non_avg_momentum + (min(max(frac, mean)-min(frac, mean), 1+min(frac, mean)-max(frac, mean)))**2
    momentum = np.sqrt(non_avg_momentum/len(list_of_frac))
    return momentum*24*3600

def compute_confidence_from_spreading(ts_in_cluster:list, high_confidence_boundary: float, low_confidence_boundary: float):
    second_momentum = compute_second_momentum(ts_in_cluster)
    confidence = -1/(low_confidence_boundary - high_confidence_boundary)*second_momentum + 1+(high_confidence_boundary)/(low_confidence_boundary-high_confidence_boundary)
    if confidence >= 1:
        confidence = 1
    elif confidence <= 0:
        confidence = 0
    return confidence


def check_for_times_during_last_x_days(current_timestamp, considered_timestamps: list, pair_of_boundaries: tuple, confidence_days=7):
    current_day = current_timestamp.floor("d")
    # Convert time object corresponding to left boundary into timedelta
    delta_min_boundary = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), pair_of_boundaries[0]) - pd.Timestamp("2000-01-01")

    # Convert time object corresponding to left boundary into timedelta
    delta_max_boundary = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), pair_of_boundaries[1]) - pd.Timestamp("2000-01-01")
    
    nr_days_in_cluster = 0
    already_considered_days = 0
    
    for i in range(50):
        if check_if_weekend(current_timestamp) == check_if_weekend(current_timestamp - (i+1)*pd.Timedelta(1, "d")):
            for window_opening_time in considered_timestamps:
                if (current_day - (i+1)*pd.Timedelta(1, "d") + delta_min_boundary <= window_opening_time and 
                    window_opening_time <= current_day - (i+1)*pd.Timedelta(1, "d") + delta_max_boundary):
                    nr_days_in_cluster += 1
                    break
            already_considered_days += 1
            if already_considered_days == confidence_days:
                break

    return nr_days_in_cluster

def compute_confidence_by_daily_apperance(current_timestamp, considered_timestamps: list, pair_of_boundaries: tuple, confidence_days=7):
    nr_days_in_cluster = check_for_times_during_last_x_days(current_timestamp, considered_timestamps, pair_of_boundaries, confidence_days=7)
    return nr_days_in_cluster/confidence_days

def check_if_weekend(current_timestamp: pd.Timestamp):
        if current_timestamp.weekday() <= 4: # Mo, Tu, Wd, Th, Fr
            return False
        else:
            return True