"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
from ..georad import georef_rdata as geo


class Rad_scan:
    """
    A Towerpy class to store radar scan data.

    Attributes
    ----------
        elev_angle : float
            Elevation angle at which the scan was taken, in deg.
        file_name : str
            Name of the file containing radar data.
        scandatetime : datetime
            Date and time of scan.
        site_name : str
            Name of the radar site.
        georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        params : dict
            Radar technical details.
        vars : dict
            Radar variables.
    """

    def __init__(self, filename, site_name=None):
        self.file_name = filename
        self.site_name = site_name

    def ppi_emptylike(self, nrays=360, ngates=425, elev=0.5,
                      rad_vars='default', scandt=None, tz='Europe/London'):
        r"""
        Create an empty object listing different radar parameters.

        Parameters
        ----------
        nrays : int
            Number of rays on the radar sweep. The default is 360.
        ngates : int, optional
            Number of bins comprising the radar rays. The default is 425.
        elev : float, optional
            Elevation angle of the radar scan. The default is 0.5.
        rad_vars : list, optional
            Polarimetric variables to add to the object. The default are:
                :math:`Z_{H} [dBZ]`, :math:`Z_{DR} [dB]`,
                :math:`\rho_{HV} [-]`, :math:`\Phi_{DP} [deg]`, :math:`V [m/s]`
                The default is 'default'.
        scandt : datetime, optional
            Date and time of the scan. If not provided, datetime.now is used.
            The default is None.
        tz : str
            Key/name of the radar data timezone. The given tz string is then
            retrieved from the ZoneInfo module. Default is 'Europe/London'
        """
        #  add aditional pol vars defined by the user
        if rad_vars == 'default':
            radvars = ['ZH [dBZ]', 'ZDR [dB]', 'PhiDP [deg]', 'rhoHV [-]',
                       'V [m/s]']
        else:
            radvars = rad_vars

        # create dicts to store the empty arrays
        # poldata = {i: np.empty([nrays, ngates],dtype=float) for i in radvars}
        poldata = {i: np.nan for i in radvars}
        parameters = {'nvars': len(radvars), 'ngates': int(ngates),
                      'nrays': int(nrays), 'gateres [m]': np.nan,
                      'rpm': np.nan, 'prf [Hz]': np.nan,
                      'pulselength [ns]': np.nan, 'avsamples': np.nan,
                      'wavelength [cm]': np.nan,
                      'latitude [dd]': np.nan, 'longitude [dd]': np.nan,
                      'altitude [m]': 0, 'easting [km]': np.nan,
                      'northing [km]': np.nan, 'radar constant [dB]': 0,
                      'elev_ang [deg]': elev,
                      'beamwidth [deg]': 1.}
        if scandt is None:
            parameters['datetime'] = dt.datetime.now(tz=ZoneInfo(tz))
            nowdt = [dt.datetime.now(tz=ZoneInfo(tz)).year,
                     dt.datetime.now(tz=ZoneInfo(tz)).month,
                     dt.datetime.now(tz=ZoneInfo(tz)).day,
                     dt.datetime.now(tz=ZoneInfo(tz)).hour,
                     dt.datetime.now(tz=ZoneInfo(tz)).minute,
                     dt.datetime.now(tz=ZoneInfo(tz)).second]
            parameters['datetimearray'] = nowdt
        else:
            parameters['datetime'] = scandt.replace(tzinfo=ZoneInfo(tz))
            udt = list(parameters['datetime'].timetuple())[: -3]
            parameters['datetimearray'] = udt
        self.vars = poldata
        self.params = parameters
        self.elev_angle = parameters['elev_ang [deg]']
        self.scandatetime = parameters['datetime']

    def ppi_create_georef(self, polarc_exist=True, elev=0.5, gate0=0,
                          gateres=250):
        """
        Create a georeferenced grid for the empty object.

        Parameters
        ----------
        polarc_exist : bool
            If True, polar coordinates (range, azimuth, elevation) are
            read directly from the georef attribute. If False,
            synthesise elevation, azimuth, and range.
            The default is True
        elev : float
            Elevation angle in degrees (used if `polarc_exist=False`).
            The default is 0.5
        gate0 : float
            Starting range gate in metres (used if `polarc_exist=False`).
            The default is 0
        gateres : float
            Gate resolution in metres (used if `polarc_exist=False`).
            The default is 250
        
        Notes
        -----
        1. Polar coordinates (azimuth, elevation) are expected to be in radians, and range values are expected to be in metres.
        2. Some radar technical details are read from Rad_scan.params
        3. This method wraps :func:`geo.ppi_georef` and updates the object's `georef` attribute with computed Cartesian grids and beam heights.
        """
        if polarc_exist:
            geogrid, _ = geo.ppi_georef(self.params, georef=self.georef)
        else:
            geogrid, geopolc = geo.ppi_georef(
                self.params, polarc_exist=False, elev=elev, gate0=gate0,
                gateres=gateres)
        # Update resolution in params
        self.params['gateres [m]'] = gateres
        if hasattr(self, 'georef'):
           self.georef.update(geogrid)
        else:
            self.georef = {'azim [rad]': geopolc['azim [rad]'],
                           'elev [rad]': geopolc['elev [rad]'],
                           'range [m]': geopolc['range [m]']}
            self.georef.update(geogrid)
