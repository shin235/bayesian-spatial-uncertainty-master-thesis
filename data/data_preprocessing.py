"""
Both datasets (covariate0605.csv, pm25_0605.csv) and the preprocessing are adapted from 
Chen et al. (2022). DeepKriging: Spatially Dependent Deep Neural Networks for Spatial Prediction.
"""

from scipy import spatial
import pandas as pd
import numpy as np


def preprocess_data():
    # Read PM2.5 data at June 05, 2019
    df1 = pd.read_csv('data/covariate0605.csv')
    df2 = pd.read_csv('data/pm25_0605.csv')

    covariates = df1.values[:, 3:]
    aqs_lonlat = df2.values[:, [1, 2]]

    # Pair the longitude and latitude on the nearest neighbor
    near = df1.values[:, [1, 2]]
    tree = spatial.KDTree(list(zip(near[:, 0].ravel(), near[:, 1].ravel())))
    tree.data
    idx = tree.query(aqs_lonlat)[1]

    df2_new = df2.assign(neighbor=idx)
    df_pm25 = df2_new.groupby('neighbor')['PM25'].mean()
    df_pm25_class = pd.cut(df_pm25, bins=[0, 12.1, 35.5], labels=["0", "1"])
    idx_new = df_pm25.index.values

    valid_idx = df_pm25_class.dropna().index
    df_pm25 = df_pm25.loc[valid_idx]
    df_pm25_class = df_pm25_class.loc[valid_idx]

    idx_new = df_pm25.index.values

    pm25 = df_pm25.values
    z = pm25[:, None]

    # print("z shape:", z.shape)

    lon = df1.values[:, 1]
    lat = df1.values[:, 2]
    normalized_lon = (lon-min(lon))/(max(lon)-min(lon))
    normalized_lat = (lat-min(lat))/(max(lat)-min(lat))
    N = lon.shape[0]
    print(f"Total gridded cells: {N}")

    num_basis = [10**2, 19**2, 37**2]
    knots_1dx = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis]
    knots_1dy = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis]
    # Wendland kernel
    basis_size = 0
    phi = np.zeros((N, sum(num_basis)))
    for res in range(len(num_basis)):
        theta = 1/np.sqrt(num_basis[res]) * 2.5
        knots_x, knots_y = np.meshgrid(knots_1dx[res], knots_1dy[res])
        knots = np.column_stack((knots_x.flatten(), knots_y.flatten()))
        for i in range(num_basis[res]):
            d = np.linalg.norm(
                np.vstack((normalized_lon, normalized_lat)).T-knots[i, :], axis=1)/theta
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi[j, i + basis_size] = (1-d[j])**6 * \
                        (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    phi[j, i + basis_size] = 0
        basis_size = basis_size + num_basis[res]

    # Remove the all-zero columns
    idx_zero = np.array([], dtype=int)
    for i in range(phi.shape[1]):
        if sum(phi[:, i] != 0) == 0:
            idx_zero = np.append(idx_zero, int(i))

    phi_reduce = np.delete(phi, idx_zero, 1)

    phi_obs = phi_reduce[idx_new, :]
    s_obs = np.vstack((normalized_lon[idx_new], normalized_lat[idx_new])).T
    X = covariates[idx_new, :]
    normalized_X = X
    for i in range(X.shape[1]):
        normalized_X[:, i] = (X[:, i]-min(X[:, i]))/(max(X[:, i])-min(X[:, i]))

    # print("Normalized covariates range:",
    #      normalized_X.min(), normalized_X.max())

    N_obs = X.shape[0]
    print(f'Observations: {N_obs}')

    X_pred = covariates.copy()
    normalized_X_pred = X_pred.copy()
    for i in range(X_pred.shape[1]):
        normalized_X_pred[:, i] = (
            X_pred[:, i]-min(X_pred[:, i]))/(max(X_pred[:, i])-min(X_pred[:, i]))

    # print("Normalized prediction input range:",
    #      normalized_X_pred.min(), normalized_X_pred.max())

    return lon, lat, idx_new, phi_obs, phi_reduce, s_obs, normalized_X, z, X_pred, normalized_X_pred
