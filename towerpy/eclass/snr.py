"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import warnings
import numpy as np
from ..datavis import rad_display


warnings.filterwarnings("ignore", category=RuntimeWarning)


class SNR_Classif:
    r"""
    A class to compute the Signal-to-Noise Ratio on radar data.

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
        min_snr : float
            Reference noise value.
        snr_class : dict
            Results of the SNR method.
        vars : dict
            Radar variables with noise removed.
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

    def signalnoiseratio(self, rad_georef, rad_params, rad_vars, min_snr=0,
                         rad_cst=None, snr_linu=False, data2correct=None,
                         classid=None, plot_method=False):
        """
        Compute the SNR and discard data using a reference noise value.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used to compute the SNR.
        min_snr : float64, optional
            Reference noise value. The default is 0.
        data2correct : dict, optional
            Variables into which noise is removed. The default is None.
        plot_method : Bool, optional
            Plot the SNR classification method. The default is False.
        """
        self.echoesID = {'pcpn': 0,
                         'noise': 3}
        if classid is not None:
            self.echoesID.update(classid)
        if rad_cst:
            rc = rad_cst
        else:
            rc = rad_params['radar constant [dB]']
        snrc = rad_vars['ZH [dBZ]'] - 20*np.log10(rad_georef['rho']) + rc
        idx = np.nonzero(snrc >= min_snr)
        snrclass = np.full(snrc.shape, np.nan)
        snrclass[idx] = 1
        snr = {'snrclass': snrclass, 'snr [dB]': snrc}
        if snr_linu is True:
            snrlu = 10 ** (0.1*snrc)
            snr['snr [linear]'] = snrlu
        if data2correct is not None:
            rdatsnr = data2correct.copy()
            for key in rdatsnr:
                rdatsnr[key] = rdatsnr[key]*snrclass
                self.vars = rdatsnr
        snr['snrclass'][np.isnan(snr['snrclass'])] = self.echoesID['noise']
        snr['snrclass'][snr['snrclass'] == 1] = self.echoesID['pcpn']
        if plot_method:
            rad_display.plot_snr(rad_georef, rad_params, snr, min_snr)
        self.min_snr = min_snr
        self.snr_class = snr

    @staticmethod
    def static_signalnoiseratio(rad_georef, rad_params, rad_vars,
                                min_snr=0, rad_cst=None, snr_linu=False,
                                data2correct=None, plot_method=False):
        """
        Compute the SNR (in dB) and discard data using a reference noise value.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used to compute the SNR..
        min_snr : float64, optional
            Reference noise value. The default is 0.
        data2correct : dict, optional
            Variables into which noise is removed. The default is None.
        plot_method : Bool, optional
            Plot the SNR classification method. The default is False.

        Returns
        -------
        Object containing the signal/noise classification.

        """
        if rad_cst:
            rc = rad_cst
        else:
            rc = rad_params['radar constant [dB]']
        snrc = rad_vars['ZH [dBZ]'] - 20*np.log10(rad_georef['rho']) + rc
        idx = np.nonzero(snrc >= min_snr)
        snrclass = np.full(snrc.shape, np.nan)
        snrclass[idx] = 1
        snr = {'snrclass': snrclass, 'snr [dB]': snrc}
        if snr_linu is True:
            snrlu = 10 ** (0.1*snrc)
            snr['snr [linear]'] = snrlu
        if data2correct is not None:
            rdatsnr = data2correct.copy()
            for key in rdatsnr:
                rdatsnr[key] = rdatsnr[key]*snrclass
                return snr, rdatsnr
        return snr
