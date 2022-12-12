from typing import (
    Tuple,
    List,
    Union,
    Dict,
    Optional,
    Any,
)
from itertools import product
from pathlib import Path
import pickle

import numpy as np
from shapely.geometry import (
    box,
    Point,
)
import geopandas as gpd


def save_pickle(data: Any, filename: Union[Path, str]) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


class GeoTransformer:

    # these are used to make shapely.geometry.box objects
    # but need to be known by multiple functions so defined as class attributes
    # as they are unlikely to change
    box_bound_x_min_name = 'minx'
    box_bound_x_max_name = 'maxx'
    box_bound_y_min_name = 'miny'
    box_bound_y_max_name = 'maxy'

    @classmethod
    def get_box_vertices(
            cls,
            centre_x_y: Union[List[float], Tuple[float, float]],
            height_of_box: float,
            x_index: int = 0
    ) -> Dict[str, float]:

        half_height = height_of_box / 2
        y_index = 1 - x_index
        return {
            cls.box_bound_x_min_name: centre_x_y[x_index] - half_height,
            cls.box_bound_y_min_name: centre_x_y[y_index] - half_height,
            cls.box_bound_x_max_name: centre_x_y[x_index] + half_height,
            cls.box_bound_y_max_name: centre_x_y[y_index] + half_height,
        }


class EuclideanEarthTransformer(GeoTransformer):

    def __init__(
            self,
            lat_min: float,
            lat_max: float,
            long_min: float,
            long_max: float,
            lat_scaler: float,
            long_scaler: float,
    ) -> None:
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.long_min = long_min
        self.long_max = long_max
        self.lat_scaler = lat_scaler
        self.long_scaler = long_scaler
        self.grid_square_length = None

        self.lat_range = self.lat_max - self.lat_min
        self.long_range = self.long_max - self.long_min

    def euclidean_to_earth_x(self, x: float) -> float:
        return ((x / self.long_scaler) * self.long_range) + self.long_min

    def euclidean_to_earth_y(self, y: float) -> float:
        return ((y / self.lat_scaler) * self.lat_range) + self.lat_min

    def euclidean_to_earth(self, x_y: Tuple[float, float]) -> Tuple[float, float]:
        """

        :param x_y:
        :return:
        """
        x_index = 0
        y_index = 1
        return self.euclidean_to_earth_y(y=x_y[y_index]), self.euclidean_to_earth_x(x=x_y[x_index])

    def earth_to_euclidean(self, lat_long: Tuple[float, float]) -> Tuple[float, float]:
        """
        This function maps WGS84 lat longs to a euclidean space
        :param lat_long: Tuple[float,float] - tuple of lat long
        :return:
        """
        lat_index = 0
        long_index = 1
        lat_euc = ((lat_long[lat_index] - self.lat_min) / self.lat_range) * self.lat_scaler
        long_euc = ((lat_long[long_index] - self.long_min) / self.long_range) * self.long_scaler

        return long_euc, lat_euc

    def euclidean_box_vertices_to_earth(self, euclidean_box_vertices: Dict[str, float]) -> Dict[str, float]:

        return {
            self.box_bound_x_min_name: self.euclidean_to_earth_x(euclidean_box_vertices[self.box_bound_x_min_name]),
            self.box_bound_y_min_name: self.euclidean_to_earth_y(euclidean_box_vertices[self.box_bound_y_min_name]),
            self.box_bound_x_max_name: self.euclidean_to_earth_x(euclidean_box_vertices[self.box_bound_x_max_name]),
            self.box_bound_y_max_name: self.euclidean_to_earth_y(euclidean_box_vertices[self.box_bound_y_max_name]),
        }

    def euclidean_centre_to_earth_box(self, centre: Tuple[float, float], height_of_box: Optional[float] = None) -> box:

        if height_of_box is None:
            height_of_box = self.grid_square_length

        if height_of_box is None:
            raise RuntimeError('must provide height of box')

        euclidean_box_vertices = self.get_box_vertices(centre_x_y=centre, height_of_box=height_of_box)
        earth_box_vertices = self.euclidean_box_vertices_to_earth(euclidean_box_vertices=euclidean_box_vertices)
        return box(**earth_box_vertices)

    def get_euclidean_grid(
            self,
            n_squares_in_x: int
    ) -> gpd.GeoDataFrame:

        # because the pre-selected bounds are rectangular rather than square the y lin space is actually smaller than
        # the x lin space by the ratio of the lengths, this allows approximate squares to be make in the Euclidean space
        n_squares_in_y = int(
            n_squares_in_x * (self.lat_scaler / self.long_scaler)
        )
        x_linspace = np.linspace(0, self.long_scaler, n_squares_in_x)
        y_linspace = np.linspace(0, self.lat_scaler, n_squares_in_y)

        self.grid_square_length = self.long_scaler / (n_squares_in_x - 1)

        centres_euc_grid = list(product(x_linspace, y_linspace))

        base_gpd = gpd.GeoDataFrame(
            geometry=[Point(i) for i in centres_euc_grid],
            data={
                'centres_euc_grid': centres_euc_grid,
            },
        ).set_crs('EPSG:3857')
        base_gpd['poly_earth'] = base_gpd['centres_euc_grid'].apply(self.euclidean_centre_to_earth_box)
        base_gpd = base_gpd.set_geometry('poly_earth')

        return base_gpd
