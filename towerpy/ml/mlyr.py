"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

from functools import reduce
from itertools import product
import numpy as np
from ..base import TowerpyError
from ..utils import radutilities as rut
from ..datavis import rad_display
from ..datavis import rad_interactive


class MeltingLayer:
    """
    A class to determine the melting layer using weather radar data.

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
    ml_top : float
        Top of the detected melting layer, in km.
    ml_bottom : float
        Bottom of the detected melting layer, in km.
    ml_thickness : float
        Thickness of the detected melting layer, in km.
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
        self.ml_source = getattr(radobj, 'profs_type',
                                 'user-defined') if radobj else 'user-defined'
        self.profs_type = getattr(radobj, 'profs_type',
                                  'user-defined') if radobj else 'user-defined'

    def findpeaksboundaries(profile_v, profile_h, param_k=None):
        """
        Find peaks inside a profile using signal processing.

        Parameters
        ----------
        profile : array
            Radar profile built from high elevation scans.
        pheight : array
            Heights correspondig to the radar profiles, in km.

        Returns
        -------
        peaks_idx : dict
            Index values of the peaks found within the profiles.

        """
        import scipy.signal as scs

        peaks, peaks_props = scs.find_peaks(profile_v, height=(None, None),
                                            threshold=(None, None),
                                            prominence=(param_k*.7, 2),
                                            width=(None, None),
                                            plateau_size=(None, None),
                                            # rel_height=0.985
                                            rel_height=0.99
                                            )
        peaks_props['left_ips'] = np.interp(
            peaks_props['left_ips'], np.arange(0, len(profile_v)),
            profile_h)
        peaks_props['right_ips'] = np.interp(
            peaks_props['right_ips'], np.arange(0, len(profile_v)),
            profile_h)

        if peaks.size == 0 or np.all(np.isnan(peaks)):
            peaks_idx = {'idxmax': np.nan, 'idxtop': np.nan, 'idxbot': np.nan,
                         'mltop': np.nan, 'mlbtm': np.nan, 'peakmaxvalue': np.nan,
                         'mlpeak': np.nan}
        else:
            bbpeak_naiveidx = np.nanargmax(peaks_props['peak_heights'])
            peakmax_props = {
                (k1 if k1.endswith('ips') else k1[:-1]): v1[bbpeak_naiveidx] 
                for k1, v1 in peaks_props.items()}
            idx_top = rut.find_nearest(profile_h, peakmax_props['right_ips'],
                                       mode='major')
            idx_bot = rut.find_nearest(profile_h, peakmax_props['left_ips'],
                                       mode='minor')
            peaks_idx = {
                'idxmax': peaks[bbpeak_naiveidx],
                # 'idxtop': peakmax_props['right_base'],
                # 'idxbot': peakmax_props['left_base'],
                'idxtop': idx_top,
                'idxbot': idx_bot,
                'peakmaxvalue': peakmax_props['peak_height'],
                'peakmaxits': peakmax_props['prominence'],
                'mlpeak': profile_h[peaks[bbpeak_naiveidx]],
                # 'mltop': profile_h[peakmax_props['right_base']],
                # 'mlbtm': profile_h[peakmax_props['left_base']],
                'mltop': profile_h[idx_top],
                'mlbtm': profile_h[idx_bot]
                }
            peaks_idx['mlthk'] = peaks_idx['mltop'] - peaks_idx['mlbtm']

        return peaks_idx

    def ml_detection(self, pol_profs, min_h=0., max_h=5., zhnorm_min=5.,
                     zhnorm_max=60., rhvnorm_min=0.85, rhvnorm_max=1.,
                     upplim_thr=0.75, param_k=0.05, param_w=0.75, comb_id=None,
                     phidp_peak='left', gradv_peak='left', plot_method=False):
        r"""
        Detect melting layer signatures within polarimetric VPs/QVPs.

        Parameters
        ----------
        pol_profs : dict
            Polarimetric profiles of radar variables.
        min_h : float, optional
            Minimum height of usable data within the polarimetric profiles.
            The default is 0.
        max_h : float, optional
            Maximum height to search for the bright band peak.
            The default is 5.
        zhnorm_min : float, optional
            Min value of :math:`Z_{H}` to use for the min-max normalisation.
            The default is 5.
        zhnorm_max : float, optional
            Max value of :math:`Z_{H}` to use for the min-max normalisation.
            The default is 60.
        rhvnorm_min : float, optional
            Min value of :math:`\rho_{HV}` to use for the min-max
            normalisation. The default is 0.85.
        rhvnorm_max : float, optional
            Max value of :math:`\rho_{HV}` to use for the min-max
            normalisation. The default is 1.
        phidp_peak : str, optional
            Direction of the peak in :math:`\Phi_{DP}` related to the ML. The
            method described in [1]_ assumes that the peak points to the left,
            (see Figure 3 in the paper) but this can be changed using this
            argument.
        gradv_peak : str, optional
            Direction of the peak in :math:`gradV` related to the ML. The
            method described in [1]_ assumes that the peak points to the left,
            (see Figure 3 in the paper) but this can be changed using this
            argument.
        param_k : float, optional
            Threshold related to the magnitude of the peak used to detect the
            ML. The default is 0.05.
        param_w : float, optional
            Weighting factor used to sharpen the peak within the profile.
            The default is 0.75.
        comb_id : int, optional
            Identifier of the combination selected for the ML detection.
            If None, the method provides all the possible combinations of
            polarimetric variables for VPs/QVPs. The default is None.
        plot_method : bool, optional
            Plot the ML detection method. The default is False.

        Notes
        -----
        1. Based on the methodology described in [1]_

        References
        ----------
        .. [1] Sanchez-Rivas, D. and Rico-Ramirez, M. A. (2021)
            "Detection of the melting level with polarimetric weather radar"
            in Atmospheric Measurement Techniques Journal, Volume 14, issue 4,
            pp. 2873–2890, 13 Apr 2021 https://doi.org/10.5194/amt-14-2873-2021

        """
        min_hidx = rut.find_nearest(pol_profs.georef['profiles_height [km]'],
                                    min_h)
        max_hidx = rut.find_nearest(pol_profs.georef['profiles_height [km]'],
                                    max_h)

        # The user shall use the combID described in the paper, thus it is
        # necessary to adjust to python indexing.
        if comb_id is not None:
            comb_idpy = comb_id-1
        else:
            comb_idpy = None

        if self.profs_type.lower() == 'vps':
            if 'ZH [dBZ]' and 'rhoHV [-]' in pol_profs.vps:
                profzh = pol_profs.vps['ZH [dBZ]'].copy()
                profrhv = pol_profs.vps['rhoHV [-]'].copy()
            else:
                raise TowerpyError(r'Profiles of $Z_H$ and $\rho_{HV}$ are '
                                   'required to run this function')
            if 'ZDR [dB]' in pol_profs.vps:
                profzdr = pol_profs.vps['ZDR [dB]'].copy()
            else:
                print(r'$Z_{DR}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profzdr = np.ones_like(profzh)
            if 'PhiDP [deg]' in pol_profs.vps:
                if phidp_peak == 'left':
                    profpdp = pol_profs.vps['PhiDP [deg]'].copy()
                elif phidp_peak == 'right':
                    profpdp = pol_profs.vps['PhiDP [deg]'].copy()
                    profpdp *= -1
            else:
                print(r'$Phi_{DP}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profpdp = np.ones_like(profzh)
            if 'gradV [dV/dh]' in pol_profs.vps:
                if gradv_peak == 'left':
                    profdvel = pol_profs.vps['gradV [dV/dh]'].copy()
                elif gradv_peak == 'right':
                    profdvel = pol_profs.vps['gradV [dV/dh]'].copy()
                    profdvel *= -1
            else:
                print('gradV [dV/dh] profile was not found. A dummy one was '
                      'built to run the method.')
                profdvel = np.ones_like(profzh)
        elif self.profs_type.lower() == 'qvps':
            if 'ZH [dBZ]' and 'rhoHV [-]' in pol_profs.qvps:
                profzh = pol_profs.qvps['ZH [dBZ]'].copy()
                profrhv = pol_profs.qvps['rhoHV [-]'].copy()
            else:
                raise TowerpyError(r'Profiles of $Z_H$ and $\rho_{HV}$ are '
                                   'required to run this function')
            if 'ZDR [dB]' in pol_profs.qvps:
                profzdr = pol_profs.qvps['ZDR [dB]'].copy()
            else:
                print(r'$Z_{DR}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profzdr = np.ones_like(profzh)
            if 'PhiDP [deg]' in pol_profs.qvps:
                if phidp_peak == 'left':
                    profpdp = pol_profs.qvps['PhiDP [deg]'].copy()
                elif phidp_peak == 'right':
                    profpdp = pol_profs.qvps['PhiDP [deg]'].copy()
                    profpdp *= -1
            else:
                print(r'$Phi_{DP}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profpdp = np.ones_like(profzh)
        elif self.profs_type.lower() == 'rd-qvps':
            if 'ZH [dBZ]' and 'rhoHV [-]' in pol_profs.rd_qvps:
                profzh = pol_profs.rd_qvps['ZH [dBZ]'].copy()
                profrhv = pol_profs.rd_qvps['rhoHV [-]'].copy()
            else:
                raise TowerpyError(r'At least $Z_H$ and $\rho_{HV}$ are '
                                   + 'required to run this function')
            if 'ZDR [dB]' in pol_profs.rd_qvps:
                profzdr = pol_profs.rd_qvps['ZDR [dB]'].copy()
            else:
                print(r'$Z_{DR}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profzdr = np.ones_like(profzh)
            if 'PhiDP [deg]' in pol_profs.rd_qvps:
                if phidp_peak == 'left':
                    profpdp = pol_profs.rd_qvps['PhiDP [deg]'].copy()
                elif phidp_peak == 'right':
                    profpdp = pol_profs.rd_qvps['PhiDP [deg]'].copy()
                    profpdp *= -1
            else:
                print(r'$Phi_{DP}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profpdp = np.ones_like(profzh)

        # Normalise ZH and rhoHV
        profzh[profzh < zhnorm_min] = zhnorm_min
        profzh[profzh > zhnorm_max] = zhnorm_max
        profzh_norm = rut.normalisenanvalues(profzh, zhnorm_min, zhnorm_max)
        profrhv[profrhv < rhvnorm_min] = rhvnorm_min
        profrhv[profrhv > rhvnorm_max] = rhvnorm_max
        profrhv_norm = rut.normalisenanvalues(
            profrhv, rhvnorm_min, rhvnorm_max)
        # Combine ZH and rhoHV (norm) to create a new profile
        profcombzh_rhv = profzh_norm[min_hidx:max_hidx]*(
            1-profrhv_norm[min_hidx:max_hidx])
        # Detect peaks within the new profile
        pkscombzh_rhv = MeltingLayer.findpeaksboundaries(
            profcombzh_rhv,
            pol_profs.georef['profiles_height [km]'][min_hidx:max_hidx],
            param_k=param_k)
        # If no peaks were found, the profile is classified as No ML signatures
        if (all(value is np.nan for value in pkscombzh_rhv.values())
           or pkscombzh_rhv['peakmaxvalue'] < param_k):
            mlyr = np.nan
            mlrand = np.nan
            combin = np.nan
            idxml_top_it1 = 0
            comb_mult = []
            comb_mult_w = []
            idxml_btm_it1 = 0
        else:
            peakcombzh_rhv = (pol_profs.georef['profiles_height [km]']
                              [min_hidx:max_hidx][pkscombzh_rhv['idxmax']])
            idxml_btm_it1 = rut.find_nearest(
                pol_profs.georef['profiles_height [km]'],
                peakcombzh_rhv-upplim_thr)
            idxml_top_it1 = rut.find_nearest(
                pol_profs.georef['profiles_height [km]'],
                peakcombzh_rhv+upplim_thr)
            if idxml_top_it1 > min_hidx:
                if self.profs_type.lower() == 'vps':
                    n = 5
                    ncomb = [1-rut.normalisenan(
                        profdvel[idxml_btm_it1:idxml_top_it1]),
                             profzh_norm[idxml_btm_it1:idxml_top_it1],
                             rut.normalisenan(
                                 profzdr[idxml_btm_it1:idxml_top_it1]),
                             1-profrhv_norm[idxml_btm_it1:idxml_top_it1],
                             1-rut.normalisenan(
                                 profpdp[idxml_btm_it1:idxml_top_it1])]
                else:
                    n = 4
                    ncomb = [profzh_norm[idxml_btm_it1:idxml_top_it1],
                             rut.normalisenan(
                                 profzdr[idxml_btm_it1:idxml_top_it1]),
                             1-profrhv_norm[idxml_btm_it1:idxml_top_it1],
                             rut.normalisenan(
                                 profpdp[idxml_btm_it1:idxml_top_it1])]
                combin = np.array(list(map(list, product([0, 1],
                                                         repeat=n)))[1:])
                comb_mult = []
                for i, j in enumerate(combin):
                    nfin4 = []
                    [idx] = np.where(combin[i] == 1)
                    for idxcomb in idx:
                        nfin = ncomb[idxcomb]
                        nfin4.append(nfin)
                    nfin5 = reduce(lambda x, y: x*y, nfin4)
                    comb_mult.append(nfin5)

                comb_mult_w = [i-(param_w*(np.gradient(np.gradient(i))))
                               for i in comb_mult]
                mlrand = [MeltingLayer.findpeaksboundaries(
                        i, pol_profs.georef['profiles_height [km]'][idxml_btm_it1:idxml_top_it1],
                        param_k=param_k)
                          for i in comb_mult_w]
                for i, j in enumerate(mlrand):
                    if mlrand[i]['peakmaxvalue'] <= param_k:
                        mlrand[i] = {k: np.nan for k in mlrand[i]}
                    if (mlrand[i]['mltop'] < min_h
                        or mlrand[i]['mltop'] > max_h
                        # or mlrand[i]['mltop'] <= 0
                        ):
                        mlrand[i] = {k: np.nan for k in mlrand[i]}
                    # if mlrand[i]['mlbtm'] <= 0:
                    #     mlrand[i]['mlbtm'] = 0
                mlrandf = [{'ml_top': n['mltop'],
                            'ml_bottom': n['mlbtm'],
                            'ml_thickness': n['mltop']-n['mlbtm'],
                            'ml_peakh': n['mlpeak'],
                            'ml_peakv': n['peakmaxvalue']}
                           for n in mlrand]
                if comb_idpy is None:
                    mlyr = mlrandf
                else:
                    mlyr = mlrandf[comb_idpy]
            else:
                mlrand = np.nan
                combin = np.nan
                mlyr = np.nan
                comb_mult = []
        if isinstance(mlrand, list) and mlrand and ~np.isnan(mlrand[7]['idxmax']):
            bb_intensity = profzh[idxml_btm_it1:idxml_top_it1][mlrand[7]['idxmax']]
            bb_peakh = pol_profs.georef['profiles_height [km]'][idxml_btm_it1:idxml_top_it1][mlrand[7]['idxmax']]
        else:
            bb_intensity = np.nan
            bb_peakh = np.nan

        if plot_method:
            rad_interactive.ml_detectionvis(
                pol_profs.georef['profiles_height [km]'], profzh_norm,
                profrhv_norm, profcombzh_rhv, pkscombzh_rhv, comb_mult,
                comb_mult_w, comb_idpy, mlrand, min_hidx, max_hidx,
                param_k, idxml_btm_it1, idxml_top_it1)
        if comb_idpy is None:
            self.ml_id = mlyr
        elif comb_idpy is not None and isinstance(mlyr, dict):
            self.ml_top = mlyr['ml_top']
            self.ml_bottom = mlyr['ml_bottom']
            self.ml_thickness = mlyr['ml_thickness']
            self.profpeakv = mlyr['ml_peakv']
            self.profpeakh = mlyr['ml_peakh']
            self.bbpeakvalue = bb_intensity
            self.bb_peakh = bb_peakh
        else:
            self.ml_top = np.nan
            self.ml_bottom = np.nan
            self.ml_thickness = np.nan
            self.profpeakv = np.nan
            self.profpeakh = np.nan
            self.bbpeakvalue = bb_intensity
            self.bb_peakh = bb_peakh
    
    def ml_ppidelimitation(self, rad_georef, rad_params, beam_cone='centre',
                           classid=None, plot_method=False):
        """
        Create a PPI depicting the limits of the melting layer.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        beam_cone : {'centre', 'top', 'bottom', 'all'}, optional
            Beam‑height descriptor to use for ML gating.
            'centre' uses ``beam_height [km]`` (default),
            'top' uses ``beamtop_height [km]``,
            'bottom' uses ``beambottom_height [km]``,
            'all' returns results for all three.
        classid : dict, optional
            Modifies the key/values of the melting layer delimitation
            (regionID). The default are the same as in regionID.
        plot_method : bool, optional
            Plot the results of the ML delimitation. The default is False.

        Attributes
        ----------
        regionID : dict
            Key/values of the rain limits:
                'rain' = 1

                'mlyr' = 2

                'solid_pcp' = 3
        """
        ml_top = self.ml_top
        ml_thickness = self.ml_thickness
        ml_bottom = self.ml_bottom
        self.regionID = {'rain': 1.,
                         'mlyr': 2.,
                         'solid_pcp': 3.}
        if classid is not None:
            self.regionID.update(classid)
        gbeam2use = (['beam_height [km]'] if beam_cone == 'centre' else 
                     ['beamtop_height [km]'] if beam_cone == 'top' else 
                     ['beambottom_height [km]'] if beam_cone == 'bottom' else 
                     ['beam_height [km]', 'beamtop_height [km]',
                      'beambottom_height [km]'])
        # =============================================================================
        mlylims = {}
        for gb2d in gbeam2use:
            if isinstance(ml_top, (int, float)):
                mlt_idx = [rut.find_nearest(nbh, ml_top)
                           for nbh in rad_georef[gb2d]]
            elif isinstance(ml_top, (np.ndarray, list, tuple)):
                mlt_idx = [rut.find_nearest(nbh, ml_top[cnt])
                           for cnt, nbh in
                           enumerate(rad_georef[gb2d])]
            if isinstance(ml_bottom, (int, float)):
                if np.isnan(ml_bottom):
                    ml_bottom = ml_top - ml_thickness
                mlb_idx = [rut.find_nearest(nbh, ml_bottom)
                           for nbh in rad_georef[gb2d]]
            elif isinstance(ml_bottom, (np.ndarray, list, tuple)):
                ml_bottom = [mlt - ml_thickness[c1]
                             if np.isnan(ml_bottom[c1]) else ml_bottom[c1]
                             for c1, mlt in enumerate(ml_top)]
                mlb_idx = [rut.find_nearest(nbh, ml_bottom[cnt])
                           for cnt, nbh in
                           enumerate(rad_georef[gb2d])]
            ashape = np.zeros((rad_params['nrays'], rad_params['ngates']))
            for cnt, azi in enumerate(ashape):
                azi[:mlb_idx[cnt]] = self.regionID['rain']
                azi[mlt_idx[cnt]:] = self.regionID['solid_pcp']
            ashape[ashape == 0] = self.regionID['mlyr']
            if gb2d == 'beam_height [km]':
                mlylims['pcp_region [HC]'] = ashape
            elif gb2d == 'beamtop_height [km]':
                mlylims['pcp_region_tb [HC]'] = ashape
            elif gb2d == 'beambottom_height [km]':
                mlylims['pcp_region_bb [HC]'] = ashape
        # =============================================================================
        # self.mlyr_limits = {'pcp_region [HC]': ashape}
        self.mlyr_limits = mlylims
        if plot_method:
            rad_display.plot_ppi(rad_georef, rad_params, self.mlyr_limits,
                                 cbticks=self.regionID,
                                 ucmap='tpylc_div_yw_gy_bu')
    
    # def ml_ppidelimitation(self, rad_georef, rad_params, beam_cone='centre',
    #                        classid=None, plot_method=False):
    #     """
    #     Create a PPI depicting the limits of the melting layer.

    #     Parameters
    #     ----------
    #     rad_georef : dict
    #         Georeferenced data containing descriptors of the azimuth, gates
    #         and beam height, amongst others.
    #     rad_params : dict
    #         Radar technical details.
    #     beam_cone : {'centre', 'top', 'bottom', 'all'}, optional
    #         Beam‑height descriptor to use for ML delimitation.
    #         'centre' uses ``beam_height [km]`` (default),
    #         'top' uses ``beamtop_height [km]``,
    #         'bottom' uses ``beambottom_height [km]``,
    #         'all' returns results for all three.
    #     classid : dict, optional
    #         Modifies the key/values of the melting layer delimitation
    #         (regionID). The default are the same as in regionID.
    #     plot_method : bool, optional
    #         Plot the results of the ML delimitation. The default is False.

    #     Attributes
    #     ----------
    #     regionID : dict
    #         Key/values of the rain limits:
    #             'rain' = 1

    #             'mlyr' = 2

    #             'solid_pcp' = 3
    #     """
    #     ml_top = self.ml_top
    #     ml_thickness = self.ml_thickness
    #     ml_bottom = self.ml_bottom
        
    #     self.regionID = {'rain': 1.,
    #                      'mlyr': 2.,
    #                      'solid_pcp': 3.}
    #     if classid is not None:
    #         self.regionID.update(classid)
        
    #     beam_map = {
    #         'centre':  ['beam_height [km]'],
    #         'top':     ['beamtop_height [km]'],
    #         'bottom':  ['beambottom_height [km]'],
    #         'all':     ['beam_height [km]', 'beamtop_height [km]',
    #                     'beambottom_height [km]']}
    #     gbeam2use = beam_map.get(beam_cone, beam_map['centre'])
        
    #     # =============================================================================
    #     nrays = rad_params['nrays']
    #     ngates = rad_params['ngates']
        
    #     # --- normalise ml_top ---
    #     if np.isscalar(ml_top):
    #         ml_top_arr = np.full(nrays, ml_top, dtype=float)
    #     else:
    #         ml_top_arr = np.asarray(ml_top, dtype=float)
    #         if ml_top_arr.shape[0] != nrays:
    #             raise ValueError("ml_top must have length nrays")
        
    #     # --- handle ml_bottom according to your rules ---
    #     if np.isscalar(ml_bottom):
    #         if np.isnan(ml_bottom):
    #             ml_bottom_arr = ml_top_arr - ml_thickness
    #         else:
    #             ml_bottom_arr = np.full(nrays, ml_bottom, dtype=float)
    #     else:
    #         ml_bottom_arr = np.asarray(ml_bottom, dtype=float)
    #         if ml_bottom_arr.shape[0] != nrays:
    #             raise ValueError("ml_bottom must have length nrays")
    #         if np.isnan(ml_bottom_arr).any():
    #             raise ValueError("ml_bottom array must not contain NaN")
        
    #     mlylims = {}
    #     rain_id  = self.regionID['rain']
    #     solid_id = self.regionID['solid_pcp']
    #     mlyr_id  = self.regionID['mlyr']
        
    #     key_map = {'beam_height [km]':      'pcp_region [HC]',
    #                'beamtop_height [km]':   'pcp_region_tb [HC]',
    #                'beambottom_height [km]':'pcp_region_bb [HC]'}
    #     gate_idx = np.arange(ngates)[None, :]
    #     for gb2d in gbeam2use:
    #         ashape = np.full((nrays, ngates), mlyr_id, dtype=float)
    #         rain_mask  = gate_idx <  ml_bottom_arr[:, None]
    #         solid_mask = gate_idx >= ml_top_arr[:, None]
    #         ashape[rain_mask]  = rain_id
    #         ashape[solid_mask] = solid_id
    #         # store result
    #         mlylims[key_map[gb2d]] = ashape
    #     # =============================================================================
    #     self.mlyr_limits = mlylims
    #     if plot_method:
    #         rad_display.plot_ppi(rad_georef, rad_params, self.mlyr_limits,
    #                              cbticks=self.regionID,
    #                              ucmap='tpylc_div_yw_gy_bu')
