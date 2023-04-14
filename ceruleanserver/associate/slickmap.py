"""
Map built on Leaflet in order to visualize slick datasets
"""

from datetime import datetime, timedelta
from glob import glob
import json
import os
import random

import geopandas as gpd
from IPython.display import display, clear_output
import movingpandas as mpd
import numpy as np
import ipyleaflet as ipyl
import ipywidgets as ipyw
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd

from utils.associate import associate_ais_to_slicks, slicks_to_curves
from utils.constants import (DATA_DIR, AIS_DIR, SLICK_DIR, TRUTH_FILE,
                             HOURS_BEFORE, NUM_TIMESTEPS, BUF_VEC)
                             
from utils.gee import get_s1_tile_layer
from utils.misc import build_time_vec, get_utm_zone
from utils.trajectory import ais_points_to_trajectories, buffer_trajectories


class SlickMap:
    """
    SlickMap: A leaflet-based map application for visualizing data and associated modeling results
    """
    def __init__(self):
        self._create_map()

        self._create_logo()
        
        self._create_vector_layers()

        self._load_dataset()
        
        self._build_layer_controls()

        self._build_data_controls()


    def _create_map(self):
        """
        Create the map object
        """
        map_kwargs = dict()
        
        map_kwargs["zoom"] = 10
        map_kwargs["zoom_control"] = False
        map_kwargs["attribution_control"] = False
        map_kwargs["scroll_wheel_zoom"] = True
        map_kwargs["no_wrap"] = True
        map_kwargs["prefer_canvas"] = True

        basemap = ipyl.basemaps.Esri.WorldImagery
        map_kwargs["basemap"] = basemap

        layout = ipyw.Layout(height='800px')

        self.map = ipyl.Map(layout=layout, **map_kwargs)

    def _create_logo(self):
        """
        Create the SkyTruth logo
        """
        logo = ipyw.Image(
            value=open("skytruth_logo.png", "rb").read(),
            format='png',
            width=80,
        )

        logo_wc = ipyl.WidgetControl(widget=logo, position='topleft')
        self.map.add_control(logo_wc)

    def _create_vector_layers(self):
        """
        Create the vector layers
        """
        def random_color(feature, **kwargs):
            # pick a random color from a categorical colormap
            cmap = plt.cm.get_cmap('tab20')
            color = matplotlib.colors.rgb2hex(cmap(random.uniform(0, 1)))
            return {
                'color': color,
            }

        self.ais_layer = ipyl.GeoJSON(
            name="AIS",
            style={
                "opacity": 0.8,
                "dashArray": "9",
                "fillOpacity": 0.1,
                "weight": 1,
            },
            point_style={
                "radius": 4,
                "weight": 1,
                "opacity": 1,
            },
            style_callback=random_color
        )

        self.truth_layer = ipyl.GeoJSON(
            name="Truth",
            style={
                "color": "white",
                "opacity": 1,
                "dashArray": "0",
                "fillOpacity": 1,
                "weight": 2
            },
        )
        
        self.slick_layer = ipyl.GeoJSON(
            name="Slick",
            style={
                "color": "red",
                "fillColor": "black",
                "opacity": 0.8,
                "fillOpacity": 0.2,
                "weight": 1,
            },
            hover_style={
                "fillColor": "red",
                "fillOpacity": 0.8,
                "opacity": 1,
                "weight": 1
            }
        )

        self.slick_curve_layer = ipyl.GeoJSON(
            name="Slick Curve",
            style={
                "color": "#fc9403",
                "opacity": 1.0,
                "weight": 1
            }
        )
        self.footprint_layer = ipyl.GeoJSON(
            name="Footprint",
            style={
                "color": "yellow",
                "opacity": 0.7,
                "fillOpacity": 0,
                "weight": 1
            }
        )
        self.s1_layer = ipyl.TileLayer(nowrap=True)

        self.map.add_layer(self.s1_layer)
        self.map.add_layer(self.footprint_layer)
        self.map.add_layer(self.ais_layer)
        self.map.add_layer(self.truth_layer)
        self.map.add_layer(self.slick_layer)
        self.map.add_layer(self.slick_curve_layer)

    def _load_dataset(self):
        """
        Load the dataset
        """
        ais_files = glob(os.path.join(AIS_DIR, '*.geojson'))
        slick_files = glob(os.path.join(SLICK_DIR, '*.geojson'))

        ais = pd.DataFrame(ais_files).rename(columns={0: 'fname'})
        slick = pd.DataFrame(slick_files).rename(columns={0: 'fname'})

        ais['basename'] = ais['fname'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        slick['basename'] = slick['fname'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

        df = ais.merge(slick, on='basename', how='inner', suffixes=('_ais', '_slick'))

        # load in truth
        truth = pd.read_csv(os.path.join(DATA_DIR, TRUTH_FILE))
        truth = truth.rename(columns=
            {
                'PID': 'basename', 
                'HITL MMSI': 'ssvid_truth',
                'Algo MMSI': 'ssvid_algo'
            }
        )
        truth = truth[truth['ssvid_truth'].notna()]

        # merge truth into dataset
        df = df.merge(truth, on='basename', how='inner')
        df = df[['basename', 'fname_ais', 'fname_slick', 'ssvid_truth', 'ssvid_algo']]

        self.df = df
        self.ctr = 0 # start at the beginning of the dataset

    def _load_sample(self, ctr: int):
        """
        Load a sample from the dataset
        """
        self.basename = self.df['basename'].iloc[ctr]
        self.ais = gpd.read_file(self.df['fname_ais'].iloc[ctr])
        self.slick = gpd.read_file(self.df['fname_slick'].iloc[ctr])
        self.ssvid_truths = self.df['ssvid_truth'].iloc[ctr].split(',')
        self.ssvid_truths = [s.strip() for s in self.ssvid_truths]
        self.model_pred = [self.df['ssvid_algo'].iloc[ctr]]*len(self.ssvid_truths)

        # extract time of collection
        start_time = self.basename.split('_')[4]
        start_time = datetime.strptime(start_time, '%Y%m%dT%H%M%S')
        self.collect_time = start_time

        # create a time vector that we will interpolate the AIS tracks on
        self.time_vec = build_time_vec(self.collect_time, HOURS_BEFORE, NUM_TIMESTEPS)

        # get the best UTM zone for this sample
        self.utm_zone = get_utm_zone(self.ais)

        # reproject
        self.ais = self.ais.to_crs(self.utm_zone)
        self.slick = self.slick.to_crs(self.utm_zone)

    def _associate_sample(self):
        """
        Associate the slick to the AIS trajectories
        """
        # get trajectories from AIS data
        self.ais_trajectories = ais_points_to_trajectories(self.ais, self.time_vec)
        if len(self.ais_trajectories) == 0:
            return

        # get curves from slick data
        self.slick_clean, self.slick_curves = slicks_to_curves(self.slick)

        # build conic buffers around the trajectories
        self.ais_gdf_buf, self.ais_weighted = buffer_trajectories(
            self.ais_trajectories, 
            BUF_VEC, 
        )

        # associate slick to AIS trajectories
        self.slick_ais = associate_ais_to_slicks(
            self.ais_trajectories, 
            self.ais_gdf_buf,
            self.ais_weighted, 
            self.slick_clean,
            self.slick_curves
        )
        
        self.slick_ais['PID'] = self.basename

    def _update_sample_map(self):
        """
        Update the map with the new sample
        """
        # get interpolated trajectories as gdf
        # this is what we will use to plot the AIS tracks
        # also get the truth trajectories if they exist
        self.ais_gdf = self.ais_trajectories.to_traj_gdf()
        self.truth_gdf = self.ais_gdf[self.ais_gdf['traj_id'].isin(self.ssvid_truths)]

        # pull S1 tile layer around this collection
        s1_url, s1_footprint = get_s1_tile_layer(self.collect_time, self.basename)
        self.s1_layer.url = s1_url
        self.s1_layer.redraw()

        # update vector layers
        self.footprint_layer.data = s1_footprint
        self.ais_layer.data = json.loads(self.ais_gdf_buf.to_crs('EPSG:4326').geometry.to_json())
        self.truth_layer.data = json.loads(self.truth_gdf.to_crs('EPSG:4326').geometry.to_json())
        self.slick_layer.data = json.loads(self.slick.to_crs('EPSG:4326').geometry.to_json())
        self.slick_curve_layer.data = json.loads(self.slick_curves.to_crs('EPSG:4326').geometry.to_json())

        # update date display
        self.date_display.value = self.basename
        
        # update truth dataframe display
        truth_df = pd.DataFrame(
            {
                'truth_id': self.ssvid_truths, 
                'model_pred': self.model_pred
            }
        )
        
        with self.truth_df_display as disp:
            clear_output()
            display("Truth IDs")
            display(truth_df)

        # update model dataframe display
        disp_df = self.slick_ais.copy()
        disp_df = disp_df.sort_values(
            [
                'slick_index', 
                'slick_size', 
                'total_score'
            ], 
            ascending=[True, False, False]
        ).groupby('slick_index')
        with self.model_df_display as disp:
            clear_output()
            ctr = 0
            for name, group in disp_df:
                if ctr == 5:
                    break
                disp_group = group[
                    [
                        'total_score', 
                        'temporal_score', 
                        'overlap_score',
                        'frechet_dist', 
                        'traj_id']
                    ].head(1)
                display(disp_group)
                ctr += 1

        # center map on the slick
        self.map.center = self.slick.dissolve().centroid.to_crs('EPSG:4326').iloc[0].coords[0][::-1]

    def _build_data_controls(self):
        """
        Build the leaflet controls for the data
        """
        def run_all_samples(b):
            # get current date as string
            date_str = datetime.now().strftime('%Y%m%d')
            out_file = f"results_{date_str}.geojson"

            self.run_all_button.button_style = 'warning'
            self.full_results = gpd.GeoDataFrame(
                columns=[
                    'PID', 
                    'slick_index', 
                    'geometry', 
                    'temporal_score', 
                    'overlap_score', 
                    'frechet_distance', 
                    'total_score',
                    'traj_id'
                ],
                crs='EPSG:4326'
            )
            for i in range(len(self.df)):
                self.ctr = i
                self.data_progress.value = self.ctr + 1
                self.data_progress.description = f"{self.ctr+1}/{len(self.df)}"
                
                # skip if already in loaded file
                if self.df.iloc[i]['basename'] in self.full_results['PID'].values:
                    continue
                else:
                    self._load_sample(self.ctr)
                    self._associate_sample()
                    if len(self.ais_trajectories) == 0:
                        continue

                    # add results to full results
                    self.full_results = pd.concat(
                        [
                            self.full_results, 
                            self.slick_ais.to_crs('EPSG:4326')
                        ]
                    )
                
            self.run_all_button.button_style = 'success'
            self.full_results.to_file(out_file, driver='GeoJSON')

        def next_sample(b):
            if self.ctr < len(self.df):
                self.ctr += 1
                self.data_progress.value = self.ctr + 1
                self.data_progress.description = f"{self.ctr+1}/{len(self.df)}"
                self._load_sample(self.ctr)
                self._associate_sample()
                self._update_sample_map()

        def prev_sample(b):
            if self.ctr > 0:
                self.ctr -= 1
                self.data_progress.value = self.ctr + 1
                self.data_progress.description = f"{self.ctr+1}/{len(self.df)}"
                self._load_sample(self.ctr)
                self._associate_sample()
                self._update_sample_map()

        def random_sample(b):
            self.ctr = np.random.randint(0, len(self.df) - 1)
            self.data_progress.value = self.ctr + 1
            self.data_progress.description = f"{self.ctr+1}/{len(self.df)}"
            self._load_sample(self.ctr)
            self._associate_sample()
            self._update_sample_map()

        def update_text(change):
            this_entry = self.df[self.df.basename == change['new']]
            if len(this_entry) == 0:
                self.date_display.value = change['old']
            else:
                self.ctr = this_entry.index[0]
                self.date_display.value = change['new']
                self.data_progress.value = self.ctr + 1
                self.data_progress.description = f"{self.ctr+1}/{len(self.df)}"
                self._load_sample(self.ctr)
                self._associate_sample()
                self._update_sample_map()

        self.run_all_button = ipyw.Button(
            description='Run All',
            disabled=False,
            button_style='info',
            tooltip='Run all samples',
            layout=ipyw.Layout(width='100px')
        )
        self.run_all_button.on_click(run_all_samples)

        self.next_button = ipyw.Button(
            description='',
            disabled=False,
            button_style='info',
            tooltip='Go to the next sample',
            icon='arrow-right',
            layout=ipyw.Layout(width='50px')
        )
        self.next_button.on_click(next_sample)

        self.prev_button = ipyw.Button(
            description='',
            disabled=False,
            button_style='info',
            tooltip='Go to the previous sample',
            icon='arrow-left',
            layout=ipyw.Layout(width='50px')
        )
        self.prev_button.on_click(prev_sample)

        self.random_button = ipyw.Button(
            description='',
            disabled=False,
            button_style='info',
            tooltip='Go to a random sample',
            icon='random',
            layout=ipyw.Layout(width='50px')
        )
        self.random_button.on_click(random_sample)

        self.data_progress = ipyw.IntProgress(
            value=self.ctr + 1,
            min=1,
            max=len(self.df),
            description=f"{self.ctr+1}/{len(self.df)}",
            bar_style='info',
            orientation='horizontal',
            layout=ipyw.Layout(width='200px')
        )

        self.date_display = ipyw.Text(layout=ipyw.Layout(width='550px'))
        self.truth_df_display = ipyw.Output(layout={'border': '1px solid black'})
        self.model_df_display = ipyw.Output(layout={'border': '1px solid black'})

        self.date_display.observe(update_text, names='value')

        self.date_control = ipyl.WidgetControl(
            widget=self.date_display,
            position='bottomright'
        )

        self.map.add_control(self.date_control)

        self.data_pane = ipyw.VBox(
            [
                self.run_all_button,
                ipyw.HBox(
                    [
                        self.random_button,
                        self.prev_button,
                        self.next_button
                    ],
                ),
                self.data_progress,
                self.truth_df_display,
                self.model_df_display
            ]
        )
        self.data_pane.layout = ipyw.Layout(align_items='flex-end')

        self.data_control = ipyl.WidgetControl(
            widget=self.data_pane,
            position='bottomright'
        )

        self.map.add_control(self.data_control)
        self._load_sample(self.ctr)
        self._associate_sample()
        self._update_sample_map()

    def _build_layer_controls(self):
        """
        Build the layer controls for the map
        """
        self.layer_control = ipyl.LayersControl(position='topright')
        self.map.add_control(self.layer_control)

        self.full_screen_control = ipyl.FullScreenControl()
        self.map.add_control(self.full_screen_control)

        self.scale_control = ipyl.ScaleControl(position='bottomleft')
        self.map.add_control(self.scale_control)
        


