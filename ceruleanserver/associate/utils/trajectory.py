"""
Utilities for working with AIS trajectories
"""

from typing import List

import movingpandas as mpd
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely.geometry
import shapely.ops


def ais_points_to_trajectories(ais: gpd.GeoDataFrame, time_vec: pd.DatetimeIndex):
    """
    Convert a set of AIS points into trajectories, grouped by ssvid
    For each trajectory, interpolate the position along a specified time vector

    Args:
        ais (gpd.GeoDataFrame): AIS points with an SSVID field
        time_vec (pd.DatetimeIndex): Time vector to interpolate to
    Returns:
        mpd.TrajectoryCollection: Trajectories with interpolated positions
    """
    ais_trajectories = list()
    for ssvid, group in ais.groupby('ssvid'):
        if len(group) > 1: # ignore single points
            # build trajectory 
            traj = mpd.Trajectory(
                df=group,
                traj_id=ssvid,
                t="timestamp"
            )

            # interpolate/extrapolate to times in time_vec
            times = list()
            positions = list()
            for t in time_vec:
                pos = traj.interpolate_position_at(t)
                times.append(t)
                positions.append(pos)

            # store as trajectory
            interpolated_traj = mpd.Trajectory(
                df=gpd.GeoDataFrame(
                    {
                        "timestamp": times,
                        "geometry": positions
                    },
                    crs=ais.crs
                ),
                traj_id=ssvid,
                t="timestamp"
            )

            ais_trajectories.append(interpolated_traj)

    return mpd.TrajectoryCollection(ais_trajectories)


def buffer_trajectories(ais: mpd.TrajectoryCollection,
                        buf_vec: np.ndarray) -> gpd.GeoDataFrame:
    """
    Build conic buffers around each trajectory
    Buffer is narrowest at the start and widest at the end
    Weight is highest at the start and lowest at the end

    Args:
        ais (mpd.TrajectoryCollection): Trajectories to buffer
        buf_vec (np.ndarray): Buffer radii, in meters
        weight_vec (np.ndarray): Weights for each buffer
    Returns:
        gpd.GeoDataFrame: Buffer polygons for every trajectory
        List[gpd.GeoDataFrame]: Corresponding weighted buffer polygons
    """
    ais_buf = list()
    ais_weighted = list()
    for traj in ais:
        # grab points
        points = traj.to_point_gdf()
        points = points.sort_values(by="timestamp", ascending=False)

        # create buffered circles at points
        ps = list()
        for idx, buffer in enumerate(buf_vec):
            ps.append(points.iloc[idx].geometry.buffer(buffer))
        
        # create convex hulls from circles
        n = range(len(ps) - 1)
        convex_hulls = [shapely.geometry.MultiPolygon([ps[i], ps[i+1]]).convex_hull for i in n]

        # weight convex hulls
        weighted = list()
        for cidx, c in enumerate(convex_hulls):
            entry = dict()
            entry['geometry'] = c
            entry['weight'] = 1.0 / (cidx + 1) # weight is the inverse of the index
            weighted.append(entry)
        weighted = gpd.GeoDataFrame(weighted, crs=traj.crs)
        ais_weighted.append(weighted)

        # create polygon from hulls
        poly = shapely.ops.unary_union(convex_hulls)
        ais_buf.append(poly)

    ais_buf = gpd.GeoDataFrame(geometry=ais_buf, crs=traj.crs)
    
    return ais_buf, ais_weighted



