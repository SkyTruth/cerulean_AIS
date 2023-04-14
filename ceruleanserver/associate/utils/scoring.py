"""
Utilities for calculating scoring metrics between AIS trajectories and oil slick detections
"""

import geopandas as gpd
import movingpandas as mpd
import shapely.geometry
import shapely.ops
from shapely import frechet_distance


def compute_frechet_distance(traj: mpd.Trajectory, curve: shapely.geometry.LineString):
    """
    Compute the frechet distance between an AIS trajectory and an oil slick curve

    Args:
        traj (mpd.Trajectory): AIS trajectory
        curve (shapely.geometry.LineString): oil slick curve

    Returns:
        float: frechet distance between traj and curve
    """
    # get the trajectory coordinates as points in descending time order from collect
    traj_gdf = traj.to_point_gdf().sort_values(by='timestamp', ascending=False)

    # take the points and put them in a linestring
    traj_line = shapely.geometry.LineString(traj_gdf.geometry)

    # get the first and last points of the slick curve
    first_point = shapely.geometry.Point(curve.coords[0])
    last_point = shapely.geometry.Point(curve.coords[-1])

    # compute the distance from these points to the start of the trajectory
    first_dist = first_point.distance(shapely.geometry.Point(traj_line.coords[0]))
    last_dist = last_point.distance(shapely.geometry.Point(traj_line.coords[0]))
    
    if last_dist < first_dist:
        # change input orientation by reversing the slick curve
        curve = shapely.geometry.LineString(list(curve.coords)[::-1])

    # for every point in the curve, find the closest trajectory point and store it off
    traj_points = list()
    for curve_point in curve.coords:
        # compute the distance between this point and every point in the trajectory
        these_distances = list()
        for traj_point in traj_line.coords:
            dist = shapely.geometry.Point(curve_point).distance(shapely.geometry.Point(traj_point))
            these_distances.append(dist)

        closest_distance = min(these_distances)
        closest_idx = these_distances.index(closest_distance)
        traj_points.append(shapely.geometry.Point(traj_line.coords[closest_idx]))

    # compute the frechet distance between the sampled trajectory curve and the slick curve
    traj_line_clip = shapely.geometry.LineString(traj_points)
    dist = frechet_distance(traj_line_clip, curve)
    
    return dist


def compute_temporal_score(weighted_traj: gpd.GeoDataFrame, slick: shapely.geometry.Polygon):
    """
    Compute the temporal score between a weighted AIS trajectory and an oil slick

    Args:
        weighted_traj (gpd.GeoDataFrame): weighted AIS trajectory, with convex hull geometries and associated weights
        slick (shapely.geometry.Polygon): oil slick polygon
    Returns:
        float: temporal score between weighted_traj and slick
    """
    # spatially join the weighted convex hulls to the slick geometry
    s_gdf = gpd.GeoDataFrame(index=[0], geometry=[slick], crs=weighted_traj.crs)
    matches = gpd.sjoin(weighted_traj, s_gdf, how='inner', predicate='intersects')
    
    temporal_score = 0.0
    if ~matches.empty:
        # take the sum of the weights of the matched convex hulls
        temporal_score = matches['weight'].sum()

    return temporal_score


def compute_overlap_score(buffered_traj: shapely.geometry.Polygon, slick: shapely.geometry.Polygon):
    """
    Compute the amount of overlap between a buffered AIS trajectory and an oil slick

    Args:
        buffered_traj (shapely.geometry.Polygon): buffered AIS trajectory created by a convex hull operation
        slick (shapely.geometry.Polygon): oil slick polygon
    Returns:
        float: overlap score between buffered_traj and slick
    """
    overlap_score = slick.intersection(buffered_traj).area / slick.area
    return overlap_score


def compute_total_score(temporal_score: float, overlap_score: float, frechet_dist: float):
    """
    Compute the total score by combining the temporal score, overlap score, and frechet distance
    The final weights were determined by a coarse grid search

    Args:
        temporal_score (float): temporal score between a weighted AIS trajectory and an oil slick
        overlap_score (float): overlap score between a buffered AIS trajectory and an oil slick
        frechet_dist (float): frechet distance between an AIS trajectory and an oil slick curve
    Returns:
        float: total weighted score between a weighted AIS trajectory and an oil slick
    """
    total_score = 0.8*temporal_score + 1.4*overlap_score + 5000.0/frechet_dist
    return total_score

