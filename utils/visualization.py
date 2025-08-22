import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle


def plot_pm25_map(lon_vals, lat_vals, values,
                  title="PM2.5 Visualization",
                  vimin=0, vmax=25, label="PM2.5 (µg/m³)",
                  cmap='jet'):
    """
    Plots PM2.5 predictions on a US grid map.
    """

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # Read US shapefile and filter to 48 contiguous states
    shapefile_path = "cb_2018_us_state_500k/cb_2018_us_state_500k.shp"
    states = gpd.read_file(shapefile_path)
    us_states = states[states["STUSPS"].isin([
        "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "IA", "ID", "IL", "IN",
        "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND",
        "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
        "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
    ])]

    outline_coords = []
    for geom in us_states.geometry:
        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
            if coords[:, 0].min() > -130 and coords[:, 1].min() > 20:
                outline_coords.append(coords)
        elif geom.geom_type == 'MultiPolygon':
            for part in geom.geoms:
                coords = np.array(part.exterior.coords)
                if coords[:, 0].min() > -130 and coords[:, 1].min() > 20:
                    outline_coords.append(coords)

    cmap_obj = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vimin, vmax=vmax)

    for cx, cy, v in zip(lon_vals, lat_vals, values):
        color = cmap_obj(norm(v))
        rect = Rectangle((cx - 0.5 / 2, cy - 0.5 / 2), 0.5, 0.5,
                         facecolor=color, edgecolor='none')
        ax.add_patch(rect)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=label)

    for coords in outline_coords:
        ax.plot(coords[:, 0], coords[:, 1], color='black', linewidth=0.4)

    ax.set_title(title)
    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(-120, -65, 10))
    ax.set_yticks(np.arange(30, 50, 5))
    ax.grid(False)
    plt.tight_layout()
    plt.show()
