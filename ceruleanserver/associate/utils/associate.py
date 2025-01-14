"""
Utilities and helper functions for the oil slick Leaflet map
"""

from datetime import datetime, timedelta
import math
from typing import List

import centerline.geometry
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.spatial.distance
import shapely.geometry
import shapely.ops
from shapely import frechet_distance

from utils.scoring import (compute_frechet_distance,
                           compute_temporal_score,
                           compute_overlap_score,
                           compute_total_score)


def associate_ais_to_slicks(ais: mpd.TrajectoryCollection, 
                            buffered: gpd.GeoDataFrame,
                            weighted: List[gpd.GeoDataFrame],
                            slick: gpd.GeoDataFrame,
                            curves: gpd.GeoDataFrame):
    """
    Measure association by computing multiple metrics between AIS trajectories and slicks

    Inputs:
        ais: TrajectoryCollection of AIS trajectories
        buffered: GeoDataFrame of buffered AIS trajectories
        weighted: list of GeoDataFrames weighted AIS trajectories
        slick: GeoDataFrame of slick detections
        curves: GeoDataFrame of slick curves
    Returns:
        GeoDataFrame of slick associations
    """
    # only consider trajectories that intersect slick detections
    ais_filt = list()
    weighted_filt = list()
    buffered_filt = list()
    for idx, t in enumerate(ais):
        w = weighted[idx]
        b = buffered.iloc[idx]

        # spatially join the weighted trajectory to the slick
        b_gdf = gpd.GeoDataFrame(index=[0], geometry=[b.geometry], crs=slick.crs)
        matches = gpd.sjoin(b_gdf, slick, how="inner", predicate="intersects")
        if matches.empty:
            continue
        else:
            ais_filt.append(t)
            weighted_filt.append(w)
            buffered_filt.append(b.geometry)

    associations = list()
    if not weighted_filt: # no associations found
        for idx in range(len(slick)):
            # return a geodataframe that has the same format as the usual case
            entry = dict()
            entry['geometry'] = slick.iloc[idx].geometry
            entry['slick_index'] = idx
            entry['slick_size'] = None
            entry['temporal_score'] = None
            entry['overlap_score'] = None
            entry['frechet_dist'] = None
            entry['total_score'] = None
            entry['traj_id'] = None
            associations.append(entry)

        associations = gpd.GeoDataFrame(associations, crs=slick.crs)
        return associations

    # create trajectory collection from filtered trajectories
    ais_filt = mpd.TrajectoryCollection(ais_filt)

    # iterate over each slick
    for idx in range(len(slick)):
        s = slick.iloc[idx]
        c = curves.iloc[idx]
        
        # iterate over filtered trajectories
        for t, w, b in zip(ais_filt, weighted_filt, buffered_filt):
            # compute temporal score
            temporal_score = compute_temporal_score(w, s.geometry)

            # compute overlap score
            overlap_score = compute_overlap_score(b, s.geometry)

            # compute frechet distance between trajectory and slick curve
            frechet_dist = compute_frechet_distance(t, c.geometry)

            # compute total score from these three metrics
            total_score = compute_total_score(temporal_score, overlap_score, frechet_dist)

            entry = dict()
            entry['geometry'] = s.geometry
            entry['slick_index'] = idx
            entry['slick_size'] = s.geometry.area
            entry['temporal_score'] = temporal_score
            entry['overlap_score'] = overlap_score
            entry['frechet_dist'] = frechet_dist
            entry['total_score'] = total_score
            entry['traj_id'] = t.id
            
            associations.append(entry)

    associations = gpd.GeoDataFrame(associations, crs=slick.crs)
    
    return associations


def slicks_to_curves(slicks: gpd.GeoDataFrame, 
                     buf_size: int = 2000, 
                     interp_dist: int = 200,
                     smoothing_factor: float = 1e9):
    """
    From a set of oil slick detections, estimate curves that go through the detections
    This process transforms a set of slick detections into LineStrings for each detection

    Inputs:
        slicks: GeoDataFrame of slick detections
        buf_size: buffer size for cleaning up slick detections
        interp_dist: interpolation distance for centerline
        smoothing_factor: smoothing factor for smoothing centerline
    Returns:
        GeoDataFrame of slick curves
    """
    # clean up the slick detections by dilation followed by erosion
    # this process can merge some polygons but not others, depending on proximity
    slicks_clean = slicks.copy()
    slicks_clean.geometry = slicks_clean.geometry.buffer(buf_size).buffer(-buf_size)

    # split slicks into individual polygons
    slicks_clean = slicks_clean.explode(ignore_index=True, index_parts=False)

    # find a centerline through detections
    slick_curves = list()
    for idx, row in slicks_clean.iterrows():
        # create centerline -> MultiLineString
        try:
            cl = centerline.geometry.Centerline(row.geometry, interpolation_distance=interp_dist)
        except:
            # sometimes the voronoi polygonization fails
            # in this case, just fit a a simple line from the start to the end
            exterior_coords = row.geometry.exterior.coords
            start_point = exterior_coords[0]
            end_point = exterior_coords[-1]
            curve = shapely.geometry.LineString([start_point, end_point])
            slick_curves.append(curve)
            continue

        # grab coordinates from centerline
        x = list()
        y = list()
        if type(cl.geometry) == shapely.geometry.MultiLineString:
            # iterate through each linestring
            for geom in cl.geometry.geoms:
                x.extend(geom.coords.xy[0])
                y.extend(geom.coords.xy[1])
        else:
            x.extend(cl.geometry.coords.xy[0])
            y.extend(cl.geometry.coords.xy[1])

        # sort coordinates in both X and Y directions
        coords = [(xc, yc) for xc, yc in zip(x, y)]
        coords_sort_x = sorted(coords, key=lambda c: c[0])
        coords_sort_y = sorted(coords, key=lambda c: c[1])

        # remove coordinate duplicates, preserving sorted order
        coords_seen_x = set()
        coords_unique_x = list()
        for c in coords_sort_x:
            if c not in coords_seen_x:
                coords_unique_x.append(c)
                coords_seen_x.add(c)
        
        coords_seen_y = set()
        coords_unique_y = list()
        for c in coords_sort_y:
            if c not in coords_seen_y:
                coords_unique_y.append(c)
                coords_seen_y.add(c)
        
        # grab x and y coordinates for spline fit
        x_fit_sort_x = [c[0] for c in coords_unique_x]
        x_fit_sort_y = [c[0] for c in coords_unique_y]
        y_fit_sort_x = [c[1] for c in coords_unique_x]
        y_fit_sort_y = [c[1] for c in coords_unique_y]

        # fit a B-spline to the centerline
        tck_sort_x, fp_sort_x, _, _ = scipy.interpolate.splrep(
            x_fit_sort_x, 
            y_fit_sort_x, 
            k=3, 
            s=smoothing_factor, 
            full_output=True
        )
        tck_sort_y, fp_sort_y, _, _ = scipy.interpolate.splrep(
            y_fit_sort_y,
            x_fit_sort_y,
            k=3, 
            s=smoothing_factor, 
            full_output=True
        )

        # choose the spline that has the lowest fit error
        if fp_sort_x <= fp_sort_y:
            tck = tck_sort_x
            x_fit = x_fit_sort_x
            y_fit = y_fit_sort_x
            
            num_points = max(round((x_fit[-1] - x_fit[0]) / 100), 5)
            x_new = np.linspace(x_fit[0], x_fit[-1], 10)
            y_new = scipy.interpolate.BSpline(*tck)(x_new)
        else:
            tck = tck_sort_y
            x_fit = x_fit_sort_y
            y_fit = y_fit_sort_y
            
            num_points = max(round((y_fit[-1] - y_fit[0]) / 100), 5)
            y_new = np.linspace(y_fit[0], y_fit[-1], num_points)
            x_new = scipy.interpolate.BSpline(*tck)(y_new)

        # store as LineString
        curve = shapely.geometry.LineString(zip(x_new, y_new))
        slick_curves.append(curve)

    slick_curves = gpd.GeoDataFrame(geometry=slick_curves, crs=slicks_clean.crs)
    return slicks_clean, slick_curves

