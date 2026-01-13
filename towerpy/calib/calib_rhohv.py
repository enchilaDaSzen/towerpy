"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import datetime as dt
import copy
import numpy as np
import xarray as xr
import xradar as xrd
from scipy.optimize import minimize_scalar
from sklearn.metrics import root_mean_squared_error as sklrmse
from ..datavis.rad_display import _plot_rhohvmethod_single, _plot_rhohvmethod_grid
from ..eclass.snr import signal2noiseratio
from ..utils.radutilities import xr_hist2d, _to_kilometers
from ..utils.radutilities import apply_correction_chain, record_provenance


class rhoHV_Calibration:
    r"""
    A class for calibrating the correlation coefficient :math:`\rho_{HV}`.

    Attributes
    ----------
    elev_angle : float
        Elevation angle at where the scan was taken, in degrees.
    file_name : str
        Name of the file containing radar data.
    scandatetime : datetime
        Date and time of scan.
    site_name : str
        Name of the radar site.
    vars : dict
        Corrected :math:`\rho_{HV}` and user-defined radar variables.
    """

    def __init__(self, radobj=None):
        self.elev_angle = getattr(radobj, 'elev_angle',
                                  None) if radobj else None
        self.file_name = getattr(radobj, 'file_name',
                                 None) if radobj else None
        self.scandatetime = getattr(radobj, 'scandatetime',
                                    None) if radobj else None
        self.site_name = getattr(radobj, 'site_name',
                                 None) if radobj else None
    
    def rhohv_noise_correction(self, rad_georef, rad_params, rad_vars,
                               mode="exp", exp_curvet=20.0, eps=0.005,
                               rhohv_theo=(0.9, 1.0), noise_level=(0, 100), 
                               bins_rho=(0.8, 1.1, 0.005), bins_snr=(5, 30, 0.1),
                               plot_method=False, data2correct=None):
        r"""
        Applies noise‑bias correction to :math:`\rho_{HV}`.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used for the correction method.
            The default is None.
        data2correct : dict, optional
            Dictionary to update the corrected :math:`\rho_{HV}`.
            The default is None.

        Notes
        -----
        1. Based on the method described in [1]_
        2. See the xarray implementation (rhohv_noisecorrection) for more details.

        References
        ----------
        .. [1] Ryzhkov, A. V.; Zrnic, D. S. (2019). Radar Polarimetry for
            Weather Observations (1st ed.). Springer International Publishing.
            https://doi.org/10.1007/978-3-030-05093-1
        """
        # =============================================================================
        # assign radar params
        # =============================================================================
        prms_mod = {k1.replace(' ', ('_')).replace('[m/s]', '[ms-1]'):
                    v1 for k1, v1 in rad_params.items()
                    if not isinstance(v1, dict) and not isinstance(v1, dt.datetime)
                    and not isinstance(v1, np.ndarray)}
        prms_mod2 = {
            f"{outer_key.replace('/', '_')}({inner_key})": value
            for outer_key, inner_dict in rad_params.items()
            if isinstance(inner_dict, dict) for inner_key, value in inner_dict.items()}
        prms_mod = prms_mod | prms_mod2
        prms_mod['datetime'] = rad_params['datetime'].strftime(
            "%Y-%m-%dT%H:%M:%S.%f_%Z%z")
        prms_mod['timestamp'] = rad_params['datetime'].timestamp()
        prms_mod['file_name'] = self.file_name
        # =============================================================================
        # assign rvars
        # =============================================================================
        sweep_vars_mapping = {'DBZH': 'ZH [dBZ]', 'ZDR': 'ZDR [dB]',
                              'PHIDP': 'PhiDP [deg]', 'RHOHV': 'rhoHV [-]',
                              'VRADH': 'V [m/s]'}
        sweep_vars_attrs_f = xrd.model.sweep_vars_mapping
        azimuth = np.rad2deg(rad_georef['azim [rad]'])
        elevation = np.rad2deg(rad_georef['elev [rad]'])
        elevation_attrs = xrd.model.get_elevation_attrs()
        rng = rad_georef['range [m]']
        azimuth_attrs = xrd.model.get_azimuth_attrs(azimuth)
        range_attrs = xrd.model.get_range_attrs(rng)
        sweep = xr.Dataset(coords=dict(azimuth=(["azimuth"], azimuth, azimuth_attrs), 
                           elevation=(["azimuth"], elevation, elevation_attrs),
                           range=(["range"], rng, range_attrs)))
        sweep = sweep.assign_coords(sweep_mode="azimuth_surveillance",
                                    longitude=rad_params['longitude [dd]'],
                                    latitude=rad_params['latitude [dd]'],
                                    altitude=rad_params['altitude [m]'],
                                    time=rad_params['aztime'])
        for k1, v1 in sweep_vars_mapping.items():
            if v1 in rad_vars.keys():
                sweep = sweep.assign({k1: (['azimuth', 'range'],
                                           rad_vars[v1])})
            else:
                print(f'{k1} not in data')
            if k1 in sweep_vars_attrs_f:
                sweep[k1].attrs = sweep_vars_attrs_f[k1]
            else:
                print(f'{k1} moment_attrs not assigned ')
        
        sweep.attrs = prms_mod
        for vv in sweep.data_vars:
            if sweep[vv].dtype == "float" or sweep[vv].dtype == "float32" or sweep[vv].dtype == "float64":
                sweep[vv].encoding = {'zlib': True, 'complevel': 6}

        rhohv_nc = rhohv_noisecorrection(
            sweep, inp_names={"Z": "DBZH", "rng": "range", "rhohv": "RHOHV"},
            rhohv_theo=rhohv_theo, mode=mode, noise_level=noise_level,
            exp_curvet=exp_curvet, eps=eps, bins_rho=bins_rho, bins_snr=bins_snr,
            preserve_original=False, data2correct=None,
            plot_method=plot_method)
        
        if data2correct is None:
            self.vars = {'rhoHV [-]': rhohv_nc.RHOHV.values}
        else:
            data2cc = copy.deepcopy(data2correct)
            data2cc.update({'rhoHV [-]': rhohv_nc.RHOHV.values})
            self.vars = data2cc
        self.noise_level_dB = rhohv_nc.attrs['noise_level_dB']


# =============================================================================
# %% xarray implementation
# =============================================================================
def _build_theo_line(snr_centers, rhohv_theo, mode="linear", exp_curvet=20.0,
                     eps=0.005):
    """Internal helper for rhohv_noisecorrection."""
    rho0, rho_inf = rhohv_theo
    if mode == "linear":
        return np.linspace(rho0, rho_inf, len(snr_centers))
    elif mode == "exp":
        s0 = snr_centers[0]
        k = np.log((rho_inf - rho0) / eps) / (exp_curvet - s0)
        return rho_inf - (rho_inf - rho0) * np.exp(-k * (snr_centers - s0))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _rmse_objective(rc, Z, rng_km, rhohv_na, bins_snr, bins_rho, rhohv_theo,
                    mode="linear", exp_curvet=20.0, eps=0.005):
    '''Internal helper for rhohv_noisecorrection.'''
    snr_db = signal2noiseratio(Z, rng_km, rc, scale="db").rename("snr_db")
    snr_lin = signal2noiseratio(Z, rng_km, rc, scale="lin").rename("snr_lin")
    rhohv_corr = (rhohv_na * (1 + 1 / snr_lin)).rename("rhohv_corr")

    snr_edges = np.arange(*bins_snr)
    rho_edges = np.arange(*bins_rho)
    hist = xr_hist2d(snr_db, rhohv_corr, snr_edges, rho_edges,
                     dim=list(snr_db.dims))

    rhohv_bin_dim = [d for d in hist.dims if d.endswith("_bin")][1]
    idx = hist.argmax(dim=rhohv_bin_dim)

    rhohv_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    histmax = xr.apply_ufunc(lambda i: rhohv_centers[i], idx,
                             vectorize=True, dask="parallelized",
                             output_dtypes=[float]).values

    snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])
    theo_line = _build_theo_line(snr_centers, rhohv_theo, mode=mode, exp_curvet=exp_curvet, eps=eps)

    return sklrmse(theo_line, histmax)

def _optimise_noise_level(Z, rng_km, rhohv_na, bins_rho=(0.8, 1.1, 0.005),
                         bins_snr=(5, 30, 0.1), rhohv_theo=(0.90, 1.),
                         noise_level=(0, 100), mode="linear", exp_curvet=20.0,
                         eps=0.005):
    '''Internal helper for rhohv_noisecorrection.'''
    def objective(rc):
        return _rmse_objective(rc, Z, rng_km, rhohv_na,
                               bins_snr, bins_rho, rhohv_theo,
                               mode=mode, exp_curvet=exp_curvet, eps=eps)
    result = minimize_scalar(objective, bounds=noise_level, method="bounded")
    return result.x, result.fun


def rhohv_noisecorrection(ds, inp_names=None, rhohv_theo=(0.9, 1.0), mode="exp",
                          exp_curvet=20.0, eps=0.005, noise_level=(0, 100),
                          bins_rho=(0.8, 1.1, 0.005), bins_snr=(5, 30, 0.1),
                          data2correct=None, preserve_original=True,
                          plot_method=False):
    r"""
    Correct noise-bias in the radar correlation coefficient (rhoHV).
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing at least:
        - reflectivity (e.g. "DBTH")
        - range (e.g. "range")
        - raw correlation coefficient rhoHV (e.g. "URHOHV")
    inp_names : dict, optional
        Mapping of variable names in `ds`. Keys: {"Z", "rng", "rhohv"}.
        Defaults: {"Z": "DBTH", "rng": "range", "rhohv": "URHOHV"}.
    bins_rho : tuple of float, optional
        rhoHV binning interval as (start, stop, step). Default is (0.8, 1.1, 0.005).
    bins_snr : tuple of float, optional
        SNR binning interval as (start, stop, step). Default is (5, 30, 0.1).
    rhohv_theo : tuple of float, optional
        Theoretical rhoHV range expected in rain (rhoHV_0, rhoHV_{inf}).
        Default is (0.90, 1.0).
    noise_level : tuple of float, optional
        Bounds for radar constant optimisation (min, max). Default is (0, 100).
    mode : {"linear", "exp", "piecewise"}, optional
        Functional form of the theoretical rhoHV–SNR curve. Default is "exp".
    exp_curvet : float, optional
        Transition point for "exp" mode. Default is 20.0.
    eps : float, optional
        Small tolerance for exponential decay. Default is 0.005.
    data2correct : xarray.Dataset, optional
        If provided, this dataset is updated with corrected rhoHV.
        If None, a new dataset is created containing `RHOHV_corr`.
    preserve_original : bool, optional
        Only applies when `data2correct` is provided:
        - True: keep raw rhoHV and add `RHOHV_corr`
        - False: overwrite raw rhoHV
    plot_method : bool, optional
        If True, plots both the optimised diagnostic plot and the
        calibration grid.

    Returns
    -------
    xarray.Dataset
        Dataset with corrected rhoHV and diagnostic attributes:
        - `noise_level_dB`
        - `objective_rmse`
        - `rhohv_theo`
        - `mode`, `exp_curvet`, `eps`
    
    Notes
    -----
    Based on the method described in [1]_.
    
    References
    ----------
    .. [1] Ryzhkov, A. V.; Zrnic, D. S. (2019).
           *Radar Polarimetry for Weather Observations* (1st ed.).
           Springer International Publishing.
           https://doi.org/10.1007/978-3-030-05093-1
    """
    defaults = {"Z": "DBTH", "rng": "range", "rhohv": "URHOHV"}
    names = {**defaults, **(inp_names or {})}
    rng_km = _to_kilometers(ds[names["rng"]]).values
    Z = ds[names["Z"]]
    rhohv_na = ds[names["rhohv"]]
    # Optimisation
    opt_noise, opt_rmse = _optimise_noise_level(
        Z, rng_km, rhohv_na, bins_rho, bins_snr, rhohv_theo, noise_level,
        mode=mode, exp_curvet=exp_curvet, eps=eps)
    # Correction
    snr_db = signal2noiseratio(Z, rng_km, opt_noise, scale="db").rename("snr_db")
    snr_lin = signal2noiseratio(Z, rng_km, opt_noise, scale="lin").rename("snr_lin")
    rhohv_corr = (rhohv_na * (1 + 1 / snr_lin)).rename("rhohv_corr")
    rhohv_final = rhohv_corr.rename("rhohv_corr")
    rhohv_final.attrs = {'standard_name': 'radar_correlation_coefficient_hv',
                         'long_name': 'Correlation coefficient HV',
                         'short_name': 'RHOHV',
                         'units': 'unitless'}
    # Histogram
    snr_edges = np.arange(*bins_snr)
    rho_edges = np.arange(*bins_rho)
    hist = xr_hist2d(snr_db, rhohv_final, snr_edges, rho_edges,
                     dim=list(snr_db.dims))
    # Extract maxima per SNR bin (bin centers)
    rhohv_bin_dim = [d for d in hist.dims if d.endswith("_bin")][1]
    idx = hist.argmax(dim=rhohv_bin_dim)

    rhohv_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    histmax = xr.apply_ufunc(lambda i: rhohv_centers[i], idx, vectorize=True,
                             dask="parallelized", output_dtypes=[float])
    # Define centers and theoretical line for plotting
    snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])
    # Theoretical line according to chosen mode
    theo_line = _build_theo_line(snr_centers, rhohv_theo,
                                 mode=mode, exp_curvet=exp_curvet, eps=eps)
    # --- output dataset ---
    suffix = "_nsc" if preserve_original else ""
    if data2correct is not None:
        # Update existing dataset
        ds_out = data2correct.copy()
        if preserve_original:
            corrected_name = f"RHOHV{suffix}"
            ds_out[corrected_name] = rhohv_final
            varname = corrected_name
        else:
            ds_out = ds_out.drop_vars(names["rhohv"])
            corrected_name = 'RHOHV'
            ds_out[corrected_name] = rhohv_final
            varname = corrected_name
    else:
        # Create new dataset with only corrected field
        ds_out = xr.Dataset()
        corrected_name = f"RHOHV{suffix}"
        ds_out[corrected_name] = rhohv_final
        varname = corrected_name
        # Copy attrs from original dataset
        ds_out.attrs = ds.attrs.copy()
    # --- attach diagnostics as attrs ---
    params = {"noise_level_dB": float(opt_noise),
              "objective_rmse": float(opt_rmse),
              "rhohv_theo": rhohv_theo,
              "mode": mode, "exp_curvet": exp_curvet, "eps": eps,
              "noise_level_bounds": noise_level}
    
    ds_out = apply_correction_chain(ds_out, varname=varname, params=params,
                                    step="noise_correction", suffix=suffix)
    ds_out.attrs.update(params)
    outputs = list(ds_out.keys())
    ds_out = record_provenance(ds_out, function="rhohv_noisecorrection",
                               inputs=ds.keys(), outputs=outputs,
                               parameters=params)    
    # Plot
    if plot_method:
        _plot_rhohvmethod_single(snr_edges, rho_edges, hist, snr_db, rhohv_na,
                     snr_centers, theo_line, histmax, opt_noise)
        _plot_rhohvmethod_grid(Z, rng_km, rhohv_na,
                   bins_snr=bins_snr, bins_rho=bins_rho,
                   rhohv_theo=rhohv_theo,
                   opt_noise=opt_noise, mode=mode, exp_curvet=exp_curvet, eps=eps)
    return ds_out
