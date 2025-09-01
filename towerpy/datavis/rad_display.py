"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from ..utils import radutilities as rut
from ..base import TowerpyError
from ..utils import unit_conversion as tpuc
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def pltparams(var2plot, rad_varskeys, vars_bounds, ucmap=None, unorm=None,
              cb_ext=None):
    """
    Create parameters for plots.

    Parameters
    ----------
    var2plot : str
        Key of the radar variable to plot. The default is None.
        This option will plot ZH or the 'first' element in the
        rad_vars dict.
    rad_varskeys : list
        List of radar variables names.
    vars_bounds : dict containing key and 3-element tuple or list
        The default are:
           {'ZH [dBZ]': [-10, 60, 15], 'ZDR [dB]': [-2, 6, 17],
            'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
            'rhoHV [-]': [0.3, .9, 1], 'AH [dB/km]': [0, 0.5, 11],
            'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
            'LDR [dB]': [-30, 10, 17], }, 'Rainfall [mm/h]': [0, 64, 11],
            'Rainfall [mm]': [0, 200, 14], 'SQI [0-1]': [0, 1, 11]
            'beam_height [km]': [0, 7, 36]}
    ucmap : colormap, optional
        User-defined colormap, either a mpl.colors.ListedColormap,
        or string from matplotlib.colormaps.
    unorm : dict containing mpl.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto
        radar data. The default is None.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    """
    lpv = {'ZH [dBZ]': [-10, 60, 15], 'ZDR [dB]': [-2, 6, 17],
           'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
           'rhoHV [-]': [0.3, .9, 1], 'AH [dB/km]': [0, 0.5, 11],
           'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1.8, 0.6, 13],
           'Rainfall [mm/h]': [0, 64, 11], 'Rainfall [mm]': [0, 200, 14],
           'beam_height [km]': [0, 7, 36], 'SQI [0-1]': [0, 1, 11]}
    if var2plot is not None:
        if var2plot == 'LDR [dB]' or 'LDR [dB]' in rad_varskeys:
            lpv['LDR [dB]'] = [-30, 10, 17]
        if var2plot == 'PIA [dB]' or 'PIA [dB]' in rad_varskeys:
            lpv['PIA [dB]'] = [0, 20, 17]
    if vars_bounds is not None:
        lpv.update(vars_bounds)
    if unorm is not None:
        lpv2 = {key: [value.vmin, value.vmax, value.N]
                for key, value in unorm.items()}
        lpv.update(lpv2)
    #
    bnd = {key[key.find('['):]: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
    if vars_bounds is None or 'Rainfall [mm/h]' not in vars_bounds.keys():
        bnd['[mm/h]'] = np.array((0.1, 1, 2, 4, 8, 12, 16, 20, 24, 30, 36, 48,
                                 56, 64))
    if vars_bounds is None or 'Rainfall [mm]' not in vars_bounds.keys():
        bnd['[mm]'] = np.array((0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                75, 100, 125, 150, 200))
    #
    cmaph = {'[dBZ]': mpl.colormaps['tpylsc_rad_ref'],
             '[-]': mpl.colormaps['tpylsc_rad_pvars'],
             '[dB]': mpl.colormaps['tpylsc_rad_2slope'],
             '[deg/km]': mpl.colormaps['tpylsc_rad_2slope'],
             '[dB/km]': mpl.colormaps['tpylsc_rad_pvars'],
             '[m/s]': mpl.colormaps['tpylsc_div_dbu_rd'],
             '[mm/h]': mpl.colormaps['tpylsc_rad_rainrt'],
             '[mm]': mpl.colormaps['tpylsc_rad_rainrt'],
             '[km]': mpl.colormaps['gist_earth'],
             '[dV/dh]': mpl.colormaps['tpylsc_rad_2slope_r'],
             }
    cmaph['[mm/h]'].set_under('whitesmoke')
    cmaph['[mm]'].set_under('whitesmoke')
    if var2plot and '[dB]' in var2plot and var2plot != 'ZDR [dB]':
        cmaph['[dB]'] = mpl.colormaps['tpylsc_rad_pvars']
    if var2plot == 'LDR [dB]':
        cmaph['[dB]'] = mpl.colormaps['tpylsc_rad_2slope_r']
    if var2plot == 'PIA [dB]':
        cmaph['[dB]'] = mpl.colormaps['tpylsc_useq_fiery']
    # elif len(rad_varskeys) == 1 and 'LDR [dB]' in rad_varskeys:
    #     cmaph['[dB]'] = mpl.colormaps['tpylsc_rad_2slope_r']
    if ucmap is not None:
        if var2plot:
            if isinstance(ucmap, str):
                cmaph[var2plot[var2plot.find('['):]] = mpl.colormaps[ucmap]
            else:
                cmaph[var2plot[var2plot.find('['):]] = ucmap
        else:
            if 'ZH [dBZ]' in rad_varskeys:
                v2pdmmy = 'ZH [dBZ]'
            else:
                v2pdmmy = list(rad_varskeys)[0]
            if isinstance(ucmap, str):
                cmaph[v2pdmmy[v2pdmmy.find('['):]] = mpl.colormaps[ucmap]
            else:
                cmaph[v2pdmmy[v2pdmmy.find('['):]] = ucmap
    #
    cmapext = {'[dBZ]': 'both', '[-]': 'both', '[dB]': 'both',
               '[deg/km]': 'both', '[m/s]': 'both', '[mm/h]': 'max',
               '[mm]': 'max', '[km]': 'max', '[dB/km]': 'max',
               '[0-1]': 'both', '[dV/dh]': 'both'}
    if var2plot == 'rhoHV [-]':
        cmapext['[-]'] = 'min'
    if cb_ext:
        cb_ext2 = {key[key.find('['):]: value for key, value in cb_ext.items()}
        cmapext.update(cb_ext2)
    #
    dnorm = {key: [value2 for key2, value2 in unorm.items()
                   if key2[key2.find('['):] == key][0]
             if unorm and key in [key2[key2.find('['):]
                                  for key2, value2 in unorm.items()]
             else
             mpc.BoundaryNorm(value, cmaph.get(
                 key[key.find('['):], mpl.colormaps['tpylsc_rad_pvars']).N,
                 extend=cmapext.get(key[key.find('['):], 'both'))
             for key, value in bnd.items()}
    #
    cbtks_fmt = 0
    tcks = None
    if var2plot is None or var2plot == 'ZH [dBZ]':
        if 'ZH [dBZ]' in rad_varskeys:
            var2plot = 'ZH [dBZ]'
            normp = dnorm['[dBZ]']
            if dnorm['[dBZ]']:
                tcks = dnorm['[dBZ]'].boundaries
            else:
                tcks = bnd['[dBZ]']
        else:
            var2plot = list(rad_varskeys)[0]
            normp = dnorm.get(var2plot[var2plot.find('['):])
            if dnorm.get(var2plot[var2plot.find('['):]):
                tcks = dnorm.get(var2plot[var2plot.find('['):]).boundaries
            else:
                tcks = bnd.get(var2plot[var2plot.find('['):])
    else:
        normp = dnorm.get(var2plot[var2plot.find('['):])
        # tcks = bnd.get(var2plot[var2plot.find('['):])
        if dnorm.get(var2plot[var2plot.find('['):]):
            tcks = dnorm.get(var2plot[var2plot.find('['):]).boundaries
        else:
            tcks = bnd.get(var2plot[var2plot.find('['):])
    if var2plot == 'rhoHV [-]':
        cbtks_fmt = 2
    if '[dB]' in var2plot:
        cbtks_fmt = 1
    if '[mm/h]' in var2plot:
        cbtks_fmt = 1
        # tickLabels = map(str, tcks)
    if '[mm]' in var2plot:
        cbtks_fmt = 1
    if '[km]' in var2plot:
        cbtks_fmt = 2
    if '[dB/km]' in var2plot:
        cbtks_fmt = 2
    if '[deg/km]' in var2plot:
        cbtks_fmt = 1
    if '[dV/dh]' in var2plot:
        cbtks_fmt = 2
    if tcks is not None and len(tcks) > 20:
        tcks = None

    return lpv, bnd, cmaph, cmapext, dnorm, var2plot, normp, cbtks_fmt, tcks


def plot_ppi(rad_georef, rad_params, rad_vars, var2plot=None, mlyr=None,
             vars_bounds=None, ucmap=None, unorm=None, plot_contourl=None,
             contour_kw=None, coord_sys='rect', cpy_feats=None, data_proj=None,
             proj_suffix='osgb', xlims=None, ylims=None, ring=None,
             range_rings=None, rd_maxrange=False, pixel_midp=False,
             points2plot=None, ptsvar2plot=None, cbticks=None, cb_ext=None,
             fig_title=None, fig_size=None, font_sizes='regular'):
    """
    Display a radar PPI scan.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars : dict
        Dict containing radar variables to plot.
    var2plot : str, optional
        Key of the radar variable to plot. The default is None. This option
        will plot ZH or look for the 'first' element in the rad_vars dict.
    mlyr : MeltingLayer Class, optional
        Plot the melting layer height. ml_top (float, int, list or np.array)
        and ml_bottom (float, int, list or np.array) must be explicitly
        defined. The default is None.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    plot_contourl: str, optional
        Key of the variable (within rad_vars) used to plot contour lines.
        Levels and normalisation are retrieved from vars_bounds, but
        these and other parameters can be overridden using the contour_kw
        parameter.
    contour_kw:
       Additional keyword arguments passed to matplotlib.pyplot.contour.
    coord_sys : 'rect' or 'polar', optional
        Coordinates system (polar or rectangular). The default is 'rect'.
    cpy_feats : dict, optional
        Cartopy attributes to add to the map. The default are:
         {'status': False, 'add_land': False, 'add_ocean': False,
         'add_coastline': False, 'add_borders': False, 'add_countries': True,
         'add_provinces': True, 'borders_ls': ':', 'add_lakes': False,
         'lakes_transparency': 0.5, 'add_rivers': False, 'tiles': False,
         'tiles_source': None, 'tiles_style': None}
    data_proj : Cartopy Coordinate Reference System object, optional
        Cartopy projection used to plot the data in a map e.g.,
        ccrs.OSGB(approx=False).
    proj_suffix : str, optional
        Suffix of the georeferenced grids used to display the data.
        The X/Y grids must exist in the rad_georef dictionary, e.g.
        'grid_osgbx--grid_osgby', 'grid_utmx--grid_utmy',
        'grid_wgs84x--grid_wgs84y', etc. The default is 'osgb'.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    ring : int or float, optional
        Plot a circle in the given distance, in km.
    range_rings : int, float, list or tuple, optional
        If int or float, plot circles at a fixed range, in km.
        If list or tuple, plot circles at the given ranges, in km.
    rd_maxrange : Bool, optional
        If True, plot the radar's maximum range coverage. Note that this arg
        won't work if a polar coordinates system is used. The default is False.
    pixel_midp : Bool, optional
        If True, mark the mid-point of all radar pixels. Note that this arg
        won't work if a polar coordinates system is used. The default is False.
    points2plot : dict, optional
        Plot a given set of points. Dict must contain the x-coord and y-coord
        in the same format as coord_sys or proj_suffix. A third element inside
        the dict can be used as the z-coord.
    ptsvar2plot : str, optional
        Key of the variable to plot. The default is None. This option
        will looks for the 'first' element in the points2plot dict.
    cbticks : dict, optional
        Modifies the default ticks' location (dict values) and labels
        (dict keys) in the colour bar.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    fig_title : str, optional
        String to show in the plot title.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    font_sizes : str, optional
        Modifies the size of the fonts in the plot. The string has to
        be one of 'regular' or 'large'.
    """
    fsizes = {'fsz_cb': 10, 'fsz_cbt': 12, 'fsz_pt': 14, 'fsz_axlb': 12,
              'fsz_axtk': 10}
    if font_sizes == 'large':
        fsizes = {k1: v1 + 4 for k1, v1 in fsizes.items()}
    #
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot, rad_vars.keys(), vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    if var2plot is None:
        var2plot = v2p
    cmapp = cmaph.get(var2plot[var2plot.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
# =============================================================================
    # dtdes0 = f"[{rad_params['site_name']}]"
    # dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # txtboxc = (0, -.09)
    # fc, ec = 'w', 'k'
# =============================================================================
    cpy_features = {'status': False,
                    # 'coastresolution': '10m',
                    'add_land': False,
                    'add_ocean': False,
                    'add_coastline': False,
                    'add_borders': False,
                    'add_countries': True,
                    'add_provinces': True,
                    'borders_ls': ':',
                    'add_lakes': False,
                    'lakes_transparency': 0.5,
                    'add_rivers': False,
                    'tiles': False,
                    'tiles_source': None,
                    'tiles_style': None,
                    'tiles_res': 8, 'alpha_tiles': 0.5, 'alpha_rad': 1
                    }
    if cpy_feats:
        cpy_features.update(cpy_feats)
    if cpy_features['status']:
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none')
        countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='10m',
            facecolor='none')
# =============================================================================
    if fig_title is None:
        if isinstance(rad_params['elev_ang [deg]'], str):
            dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
        else:
            dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{1}f} deg. -- "
        dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
        ptitle = dtdes1 + dtdes2
    else:
        ptitle = fig_title
    plotunits = [i[i.find('['):]
                 for i in rad_vars.keys() if var2plot == i][0]
# =============================================================================
    if coord_sys == 'polar':
        if fig_size is None:
            fig_size = (6, 6.15)
        fig, ax1 = plt.subplots(figsize=fig_size,
                                subplot_kw=dict(projection='polar'))
        mappable = ax1.pcolormesh(rad_georef['theta'], rad_georef['rho'],
                                  np.flipud(rad_vars[var2plot]),
                                  shading='auto', cmap=cmapp, norm=normp)
        ax1.set_title(f'{ptitle} \n' + f'PPI {var2plot}',
                      fontsize=fsizes['fsz_pt'])
        ax1.grid(color='gray', linestyle=':')
        ax1.set_theta_zero_location('N')
        ax1.tick_params(axis='both', labelsize=fsizes['fsz_axlb'])
        ax1.set_yticklabels([])
        ax1.set_thetagrids(np.arange(0, 360, 90))
        ax1.axes.set_aspect('equal')
        if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
           or '[mm/h]' in var2plot):
            cb1 = plt.colorbar(mappable, ax=ax1, aspect=8, shrink=0.65,
                               pad=.1, norm=normp, ticks=tcks,
                               format=f'%.{cbtks_fmt}f')
            cb1.ax.tick_params(direction='in', axis='both',
                               labelsize=fsizes['fsz_cb'])
        else:
            cb1 = plt.colorbar(mappable, ax=ax1, aspect=8, shrink=0.65,
                               pad=.1, norm=normp)
            cb1.ax.tick_params(direction='in', axis='both',
                               labelsize=fsizes['fsz_cb'])
        cb1.ax.set_title(f'{plotunits}', fontsize=fsizes['fsz_cbt'])
        if cbticks is not None:
            cb1.set_ticks(ticks=list(cbticks.values()),
                          labels=list(cbticks.keys()))
        # ax1.annotate('| Created using Towerpy |', xy=txtboxc,
        #              fontsize=8, xycoords='axes fraction',
        #              va='center', ha='center',
        #              bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
        plt.tight_layout()
        # plt.show()

    elif coord_sys == 'rect' and cpy_features['status'] is False:
        # =====================================================================
        # ptitle = dtdes1 + dtdes2
        # =====================================================================
        if fig_size is None:
            fig_size = (6, 6.75)
        fig, ax1 = plt.subplots(figsize=fig_size)
        mappable = ax1.pcolormesh(rad_georef['grid_rectx'],
                                  rad_georef['grid_recty'],
                                  rad_vars[var2plot], shading='auto',
                                  cmap=cmapp, norm=normp)
        if rd_maxrange:
            ax1.plot(rad_georef['grid_rectx'][:, -1],
                     rad_georef['grid_recty'][:, -1], 'gray')
        if pixel_midp:
            binx = rad_georef['grid_rectx'].ravel()
            biny = rad_georef['grid_recty'].ravel()
            ax1.scatter(binx, biny, c='grey', marker='+', alpha=0.2)
# =============================================================================
        if points2plot is not None:
            if len(points2plot) == 2:
                ax1.scatter(points2plot['grid_rectx'],
                            points2plot['grid_recty'], color='k',
                            marker='o', )
            elif len(points2plot) >= 3:
                ax1.scatter(points2plot['grid_rectx'],
                            points2plot['grid_recty'], marker='o',
                            norm=normp, edgecolors='k',
                            c=[points2plot[ptsvar2plot]], cmap=cmapp)
# =============================================================================
        if mlyr is not None:
            if isinstance(mlyr.ml_top, (int, float)):
                mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top)
                           for nbh in rad_georef['beam_height [km]']]
            elif isinstance(mlyr.ml_top, (np.ndarray, list, tuple)):
                mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top[cnt])
                           for cnt, nbh in
                           enumerate(rad_georef['beam_height [km]'])]
            if isinstance(mlyr.ml_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom)
                           for nbh in rad_georef['beam_height [km]']]
            elif isinstance(mlyr.ml_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom[cnt])
                           for cnt, nbh in
                           enumerate(rad_georef['beam_height [km]'])]
            mlt_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                                 for cnt, ix in enumerate(mlt_idx)])
            mlt_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                                 for cnt, ix in enumerate(mlt_idx)])
            mlb_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                                 for cnt, ix in enumerate(mlb_idx)])
            mlb_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                                 for cnt, ix in enumerate(mlb_idx)])
            ax1.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(T)}$')
            ax1.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(B)}$')
            first_legend = ax1.legend(loc='upper left')
            # Add the legend manually to the Axes.
            ax1.add_artist(first_legend)
# =============================================================================
        if range_rings is not None:
            if isinstance(range_rings, range):
                range_rings = list(range_rings)
            if isinstance(range_rings, (int, float)):
                nrings = np.arange(range_rings*1000,
                                   rad_georef['range [m]'][-1],
                                   range_rings*1000)
            elif isinstance(range_rings, (np.ndarray, list, tuple)):
                nrings = np.array(range_rings) * 1000
            idx_rs = [rut.find_nearest(rad_georef['range [m]'], r)
                      for r in nrings]
            dmmy_rsx = np.array([rad_georef['grid_rectx'][:, i]
                                 for i in idx_rs])
            dmmy_rsy = np.array([rad_georef['grid_recty'][:, i]
                                 for i in idx_rs])
            dmmy_rsz = np.array([np.ones(i.shape) for i in dmmy_rsx])
            ax1.scatter(dmmy_rsx, dmmy_rsy, dmmy_rsz, c='grey', ls='--',
                        alpha=3/4)
            ax1.axhline(0, c='grey', ls='--', alpha=3/4)
            ax1.axvline(0, c='grey', ls='--', alpha=3/4)
            ax1.grid(True)
# =============================================================================
        if ring is not None:
            idx_rr = rut.find_nearest(rad_georef['range [m]'],
                                      ring*1000)
            dmmy_rx = rad_georef['grid_rectx'][:, idx_rr]
            dmmy_ry = rad_georef['grid_recty'][:, idx_rr]
            dmmy_rz = np.ones(dmmy_rx.shape)
            ax1.scatter(dmmy_rx, dmmy_ry, dmmy_rz, c='k', ls='--', alpha=3/4)
# =============================================================================
        if plot_contourl:
            ckw = {'alpha': 0.5, 'zorder': 2, 'colors': None,
                   'levels': bnd.get(plot_contourl[plot_contourl.find('['):]),
                   'norm': dnorm.get(plot_contourl[plot_contourl.find('['):]),
                   'cmap': cmaph.get(plot_contourl[plot_contourl.find('['):]),
                   'legend': False,
                   }
            if contour_kw is not None:
                ckw.update(contour_kw)
            contourlp = ax1.contour(
                rad_georef['grid_rectx'], rad_georef['grid_recty'],
                rad_vars[plot_contourl],
                **ckw)
            ax1.clabel(contourlp, inline=True, fontsize=fsizes['fsz_cbt'])
            if ckw['legend']:
                cspl, labels = contourlp.legend_elements()
                labels = [lb.replace('x = ', '') for lb in labels]
                ax1.legend(cspl, labels, title=plot_contourl,
                           loc='upper right').set_zorder(5)
# =============================================================================
        ax1_divider = make_axes_locatable(ax1)
        cax1 = ax1_divider.append_axes('top', size="7%", pad="2%")
        if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
           or '[mm/h]' in var2plot):
            cb1 = fig.colorbar(mappable, cax=cax1, orientation='horizontal',
                               ticks=tcks, format=f'%.{cbtks_fmt}f')
            cb1.ax.tick_params(direction='in', labelsize=fsizes['fsz_cb'],
                               rotation=(45 if font_sizes == 'large' else 0))
        else:
            cb1 = fig.colorbar(mappable, cax=cax1, orientation='horizontal')
            cb1.ax.tick_params(direction='in', labelsize=fsizes['fsz_cb'])
        fig.suptitle(f'{ptitle} \n' + f'PPI {var2plot}',
                     fontsize=fsizes['fsz_pt'])
        cax1.xaxis.set_ticks_position('top')
        if xlims is not None:
            ax1.set_xlim(xlims)
        if ylims is not None:
            ax1.set_ylim(ylims)
        ax1.set_xlabel('Distance from the radar [km]',
                       fontsize=fsizes['fsz_axlb'], labelpad=10)
        ax1.set_ylabel('Distance from the radar [km]',
                       fontsize=fsizes['fsz_axlb'], labelpad=10)
        ax1.tick_params(direction='in', axis='both',
                        labelsize=fsizes['fsz_axtk'])
        if cbticks is not None:
            cb1.set_ticks(ticks=list(cbticks.values()),
                          labels=list(cbticks.keys()), ha='right')
        ax1.axes.set_aspect('equal')
        # ax1.annotate('| Created using Towerpy |', xy=txtboxc,
        #              fontsize=8, xycoords='axes fraction',
        #              va='center', ha='center',
        #              bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
        # ax1.grid(True)
        plt.tight_layout()
        # plt.show()

    elif coord_sys == 'rect' and cpy_features['status']:
        # ptitle = dtdes1 + dtdes2
        proj = ccrs.PlateCarree()
        if fig_size is None:
            fig_size = (12, 6)
        if data_proj:
            proj2 = data_proj
        else:
            raise TowerpyError('User must specify the projected coordinate'
                               ' system of the radar data e.g.'
                               ' ccrs.OSGB(approx=False) or ccrs.UTM(zone=32)')
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        plt.subplots_adjust(left=0.05, right=0.99, top=0.981, bottom=0.019,
                            wspace=0, hspace=1)
        ax1 = fig.add_subplot(projection=proj)
        if xlims and ylims:
            extx = xlims
            exty = ylims
            ax1.set_extent(extx+exty, crs=proj)
        if cpy_features['tiles']:
            if (cpy_features['tiles_source'] is None
               or cpy_features['tiles_source'] == 'OSM'):
                imtiles = cimgt.OSM()
                ax1.add_image(imtiles, cpy_features['tiles_res'],
                              interpolation='spline36',
                              alpha=cpy_features['alpha_tiles'])
            elif cpy_features['tiles_source'] == 'GoogleTiles':
                if cpy_features['tiles_style'] is None:
                    imtiles = cimgt.GoogleTiles(style='street')
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
                else:
                    imtiles = cimgt.GoogleTiles(
                        style=cpy_features['tiles_style'])
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  # interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
            elif cpy_features['tiles_source'] == 'QuadtreeTiles':
                if cpy_features['tiles_style'] is None:
                    imtiles = cimgt.QuadtreeTiles()
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
            elif cpy_features['tiles_source'] == 'Stamen':
                if cpy_features['tiles_style'] is None:
                    imtiles = cimgt.Stamen(style='toner')
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
                else:
                    imtiles = cimgt.Stamen(style=cpy_features['tiles_style'])
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
        if cpy_features['add_land']:
            ax1.add_feature(cfeature.LAND)
        if cpy_features['add_ocean']:
            ax1.add_feature(cfeature.OCEAN)
        if cpy_features['add_coastline']:
            ax1.add_feature(cfeature.COASTLINE)
        if cpy_features['add_borders']:
            ax1.add_feature(cfeature.BORDERS,
                            linestyle=cpy_features['borders_ls'])
        if cpy_features['add_lakes']:
            ax1.add_feature(cfeature.LAKES,
                            alpha=cpy_features['lakes_transparency'])
        if cpy_features['add_rivers']:
            ax1.add_feature(cfeature.RIVERS)
        if cpy_features['add_countries']:
            ax1.add_feature(states_provinces, edgecolor='black', ls=":")
        if cpy_features['add_provinces']:
            ax1.add_feature(countries, edgecolor='black', )

        data_source = 'Natural Earth'
        data_license = 'public domain'
        # Add a text annotation for the license information to the
        # the bottom right corner.
        # text = AnchoredText(r'$\copyright$ {}; license: {}'
        #                     ''.format(SOURCE, LICENSE),
        #                     loc=4, prop={'size': 12}, frameon=True)
        # ax1.add_artist(text)
        print('\N{COPYRIGHT SIGN}' + f'{data_source}; license: {data_license}')
        if cpy_features['tiles_source'] == 'Stamen':
            print('\N{COPYRIGHT SIGN}' + 'Map tiles by Stamen Design, '
                  + 'under CC BY 3.0. Data by OpenStreetMap, under ODbL.')
        gl = ax1.gridlines(draw_labels=True, dms=False,
                           x_inline=False, y_inline=False)
        gl.xlabel_style = {'size': fsizes['fsz_axlb']}
        gl.ylabel_style = {'size': fsizes['fsz_axlb']}
        ax1.set_title(f'{ptitle} \n' + f'PPI {var2plot}',
                      fontsize=fsizes['fsz_pt'])
        # lon_formatter = LongitudeFormatter(number_format='.4f',
        #                                 degree_symbol='',
        #                                dateline_direction_label=True)
        # lat_formatter = LatitudeFormatter(number_format='.0f',
        #                                    degree_symbol=''
        #                                   )
        mappable = ax1.pcolormesh(rad_georef[f'grid_{proj_suffix}x'],
                                  rad_georef[f'grid_{proj_suffix}y'],
                                  rad_vars[var2plot], transform=proj2,
                                  shading='auto', cmap=cmapp, norm=normp,
                                  alpha=cpy_features['alpha_rad'])
        # ax1.xaxis.set_major_formatter(lon_formatter)
        # ax1.yaxis.set_major_formatter(lat_formatter)
        if pixel_midp:
            binx = rad_georef[f'grid_{proj_suffix}x'].ravel()
            biny = rad_georef[f'grid_{proj_suffix}y'].ravel()
            ax1.scatter(binx, biny, c='grey', marker='+', transform=proj2,
                        alpha=0.2)
        if rd_maxrange:
            ax1.plot(rad_georef[f'grid_{proj_suffix}x'][:, -1],
                     rad_georef[f'grid_{proj_suffix}y'][:, -1],
                     'gray', transform=proj2)
# =============================================================================
        if points2plot is not None:
            if len(points2plot) == 2:
                ax1.scatter(points2plot[f'grid_{proj_suffix}x'],
                            points2plot[f'grid_{proj_suffix}y'], color='k',
                            marker='o', )
            elif len(points2plot) >= 3:
                ax1.scatter(points2plot[f'grid_{proj_suffix}x'],
                            points2plot[f'grid_{proj_suffix}y'],
                            marker='o', norm=normp, edgecolors='k',
                            c=[points2plot[ptsvar2plot]], cmap=cmapp)

# =============================================================================
        def make_colorbar(ax1, mappable, **kwargs):
            ax1_divider = make_axes_locatable(ax1)
            orientation = kwargs.pop('orientation', 'vertical')
            if orientation == 'vertical':
                loc = 'right'
            elif orientation == 'horizontal':
                loc = 'top'
# =============================================================================
            ticks = tcks
            if var2plot in lpv.keys():
                if ticks is not None and len(tcks) > 20:
                    ticks = tcks[::5]
            else:
                None
# =============================================================================
            cax = ax1_divider.append_axes(loc, '7%', pad='15%',
                                          axes_class=plt.Axes)
            if cbticks is not None:
                ax1.get_figure().colorbar(
                    mappable, cax=cax, orientation=orientation,
                    ticks=list(cbticks.values()),
                    format=mticker.FixedFormatter(list(cbticks.keys())))
            else:
                ax1.get_figure().colorbar(mappable, cax=cax,
                                          orientation=orientation,
                                          ticks=ticks,
                                          format=f'%.{cbtks_fmt}f')
            cax.tick_params(direction='in', labelsize=fsizes['fsz_cb'])
            cax.xaxis.set_ticks_position('top')
            cax.set_title(plotunits, fontsize=fsizes['fsz_cbt'])
        make_colorbar(ax1, mappable, orientation='vertical')


def plot_setppi(rad_georef, rad_params, rad_vars, mlyr=None, vars_bounds=None,
                ucmap=None, unorm=None, cb_ext=None, xlims=None, ylims=None,
                ncols=None, nrows=None, fig_title=None, fig_size=None):
    """
    Plot a set of PPIs of polarimetric variables.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars : dict
        Radar variables to be plotted.
    mlyr : MeltingLayer Class, optional
        Plot the melting layer height. ml_top (float, int, list or np.array)
        and ml_bottom (float, int, list or np.array) must be explicitly
        defined. The default is None.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    ncols : int, optional
        Number of columns used to build the grid. The default is None.
    nrows : int, optional
        Number of rows used to build the grid. The default is None.
    fig_title : str, optional
        Modify the default plot title.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    if mlyr is not None:
        if isinstance(mlyr.ml_top, (int, float)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_top, (np.ndarray, list, tuple)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        if isinstance(mlyr.ml_bottom, (int, float)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_bottom, (np.ndarray, list, tuple)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        mlt_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlt_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlb_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])
        mlb_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])
    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = rad_params['elev_ang [deg]']
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{1}f} deg"
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    if fig_title is None:
        ptitle = (f"{rad_params['site_name'].title()} "
                  + f"[{dtdes1}] -- {dtdes2}")
    else:
        ptitle = fig_title
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # fc, ec = 'w', 'k'
    if nrows is None and ncols is None:
        if len(rad_vars) <= 3:
            nrw = 1
            ncl = int(len(rad_vars))
        elif len(rad_vars) > 3 and len(rad_vars) < 10 and len(rad_vars) % 2:
            ncl = int(np.ceil(len(rad_vars)/2))
            nrw = int(np.ceil(len(rad_vars)/ncl))
        elif len(rad_vars) >= 10 and len(rad_vars) % 2:
            ncl = int(np.ceil(len(rad_vars)/4))
            nrw = int(np.ceil(len(rad_vars)/ncl))
        else:
            ncl = int(np.ceil(len(rad_vars)/2))
            nrw = int(np.ceil(len(rad_vars)/ncl))
    elif nrows is not None and ncols is None:
        if len(rad_vars) <= 3:
            nrw = nrows
            ncl = int(len(rad_vars))
        else:
            nrw = nrows
            ncl = int(np.ceil(len(rad_vars)/nrw))
    elif ncols is not None and nrows is None:
        if len(rad_vars) <= 3:
            nrw = 1
            ncl = ncols
        else:
            ncl = ncols
            nrw = int(np.ceil(len(rad_vars)/ncl))
    else:
        nrw = nrows
        ncl = ncols
        if nrw * ncl < len(rad_vars):
            print('Warning: Due to the selected grid, some variables may not '
                  + 'be displayed. Please adjust your settings to view all '
                  + 'available variables.')
    if fig_size is None and nrw != 1:
        fig_size = (16, 9)
    if fig_size is None and nrw == 1:
        fig_size = (16, 4.5)
    f, ax = plt.subplots(nrw, ncl, sharex=True, sharey=True, figsize=fig_size)
    f.suptitle(f'{ptitle}', fontsize=16)
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        None, rad_vars.keys(), vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    lpv_vars = [rkey[:rkey.find('[')-1] for rkey in lpv.keys()]
    for a, (rkey, var2plot) in zip(ax.flatten(), rad_vars.items()):
        rkey_units = rkey[rkey.find('['):]
        rkey_var = rkey[:rkey.find('[')-1]
        if rkey in lpv or rkey_var in lpv_vars or [rk for rk in lpv_vars
                                                   if rkey_var.startswith(rk)]:
            lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
                rkey, rad_vars.keys(), vars_bounds, ucmap=ucmap, unorm=unorm,
                cb_ext=cb_ext)
        else:
            lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
                rkey, rad_vars.keys(), vars_bounds, cb_ext=cb_ext)
            b1 = np.linspace(np.nanmin(var2plot), np.nanmax(var2plot), 11)
            normp = mpc.BoundaryNorm(
                b1, mpl.colormaps['tpylsc_rad_pvars'].N,
                extend=cmapext.get(rkey[rkey.find('['):], 'both'))
        cmapp = cmaph.get(rkey[rkey.find('['):],
                          mpl.colormaps['tpylsc_rad_pvars'])
        # if rkey.lower().startswith('z') and '[dBZ]' in rkey:
        #     normp = dnorm.get('[dBZ]')
        #     cmapp = mpl.colormaps['tpylsc_rad_ref']
        # if rkey.lower().startswith('zdr') and '[dB]' in rkey:
        #     normp = dnorm.get('[dB]')
        #     cmapp = mpl.colormaps['tpylsc_rad_2slope']
        # if '[0-1]' in rkey:
        #     normp = dnorm.get('[0-1]')
        # if rkey == 'rhoHV [-]':
        #     norm = [mpc.BoundaryNorm(
        #         value, cmaph.get(key[key.find('['):],
        #                          mpl.colormaps['tpylsc_rad_pvars']).N,
        #         extend='min')
        #         for key, value in bnd.items() if key == '[-]'][0]
        # if rkey == 'PIA [dB]':
        #     cmapp = mpl.colormaps['tpylsc_useq_fiery']
        f1 = a.pcolormesh(rad_georef['grid_rectx'], rad_georef['grid_recty'],
                          var2plot, shading='auto', cmap=cmapp, norm=normp)
        if mlyr is not None:
            a.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                   path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                 pe.Normal()], label=r'$MLyr_{(T)}$')
            a.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                   path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                 pe.Normal()], label=r'$MLyr_{(B)}$')
            a.legend(loc='upper left')
        if xlims is not None:
            a.set_xlim(xlims)
        if ylims is not None:
            a.set_ylim(ylims)
        # a.set_title(f'{dtdes}' "\n" f'{key}')
        a.set_title(f'{rkey}', fontsize=12)
        if nrw == 1:
            a.set_xlabel('Distance from the radar [km]', fontsize=12)
        elif ncl == 1:
            a.set_ylabel('Distance from the radar [km]', fontsize=12)
        else:
            a.set_xlabel(None, size=12)
            a.set_ylabel(None, size=12)
        a.grid(True)
        a.axes.set_aspect('equal')
        a.tick_params(axis='both', which='major', labelsize=10)
        if rkey.startswith('rhoHV'):
            f.colorbar(f1, ax=a, ticks=tcks, format=f'%.{cbtks_fmt}f')
        else:
            f.colorbar(f1, ax=a)
    if nrw*ncl > len(rad_vars):
        for empax in range(nrw*ncl-len(rad_vars)):
            f.delaxes(ax.flatten()[-empax-1])
    if ax.ndim > 1:
        plt.setp(ax[-1, :], xlabel='Distance from the radar [km]')
        plt.setp(ax[:, 0], ylabel='Distance from the radar [km]')
    if nrw == 1:
        ax[0].set_ylabel('Distance from the radar [km]', fontsize=12)
    elif ncl == 1:
        ax[-1].set_xlabel('Distance from the radar [km]', fontsize=12)
    # txtboxc = (1.025, -.10)
    # txtboxc = (-3., -.10)
    # a.annotate('| Created using Towerpy |', xy=txtboxc, fontsize=8,
    #            xycoords='axes fraction', va='center', ha='center',
    #            bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.tight_layout()
    # plt.show()


def plot_mgrid(rscans_georef, rscans_params, rscans_vars, var2plot=None,
               vars_bounds=None, ucmap=None, unorm=None, cb_ext=None,
               coord_sys='rect', cpy_feats=None, proj_suffix='osgb',
               data_proj=None, xlims=None, ylims=None, ncols=None, nrows=None,
               fig_size=None):
    """
    Graph multiple PPI scans into a grid.

    Parameters
    ----------
    rscans_georef : list
        List of georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others, corresponding to each PPI scan.
    rscans_params : list
        List of radar technical details corresponding to each PPI scan.
    rscans_vars : list
        List of Dicts containing radar variables to plot corresponding to each
        PPI scan.
    var2plot : str, optional
        Key of the radar variable to plot. The default is None. This option
        will plot ZH or the 'first' element in the rad_vars dict.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped. The default are:
            {'ZH [dBZ]': [-10, 60, 15],
             'ZDR [dB]': [-2, 6, 17],
             'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
             'rhoHV [-]': [0.3, .9, 1],
             'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
             'LDR [dB]': [-35, 0, 11],
             'Rainfall [mm/h]': [0.1, 64, 11]}
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    coord_sys : 'rect' or 'polar', optional
        Coordinates system (polar or rectangular). The default is 'rect'.
    cpy_feats : dict, optional
        Cartopy attributes to add to the map. The default are:
        {
        'status': False,
        'add_land': False,
        'add_ocean': False,
        'add_coastline': False,
        'add_borders': False,
        'add_countries': True,
        'add_provinces': True,
        'borders_ls': ':',
        'add_lakes': False,
        'lakes_transparency': 0.5,
        'add_rivers': False,
        'tiles': False,
        'tiles_source': None,
        'tiles_style': None,
        }
    proj_suffix : str
        Suffix of the georeferenced grids used to display the data.
        The X/Y grids must exist in the rad_georef dictionary, e.g.
        'grid_osgbx--grid_osgby', 'grid_utmx--grid_utmy',
        'grid_wgs84x--grid_wgs84y', etc. The default is 'osgb'.
    data_proj : Cartopy Coordinate Reference System object, optional
        Cartopy projection used to plot the data in a map e.g.,
        ccrs.OSGB(approx=False).
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    ncols : int, optional
        Set the number of columns used to build the grid. The default is None.
    nrows : int, optional
        Set the number of rows used to build the grid. The default is None.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    from mpl_toolkits.axes_grid1 import ImageGrid
    from cartopy.mpl.geoaxes import GeoAxes

    dskeys = [k for i in rscans_vars for k in i.keys()]
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot,
        ('ZH [dBZ]' if all('ZH [dBZ]' in i.keys() for i in rscans_vars)
         else list(set([x for x in dskeys
                        if dskeys.count(x) >= len(rscans_vars)
                        and '[' in x]))),
        vars_bounds, ucmap=ucmap, unorm=unorm, cb_ext=cb_ext)
    if var2plot is None:
        var2plot = v2p
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # txtboxc = (0, -.09)
    # fc, ec = 'w', 'k'
    cmapp = cmaph.get(var2plot[var2plot.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
    cpy_features = {'status': False,
                    # 'coastresolution': '10m',
                    'add_land': False,
                    'add_ocean': False,
                    'add_coastline': False,
                    'add_borders': False,
                    'add_countries': True,
                    'add_provinces': True,
                    'borders_ls': ':',
                    'add_lakes': False,
                    'lakes_transparency': 0.5,
                    'add_rivers': False,
                    'tiles': False,
                    'tiles_source': None,
                    'tiles_style': None,
                    'tiles_res': 8, 'alpha_tiles': 0.5, 'alpha_rad': 1
                    }
    if cpy_feats:
        cpy_features.update(cpy_feats)
    if cpy_features['status']:
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none')
        countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='10m',
            facecolor='none')
    # TODO add fig_title
    pttl = [f"{p['elev_ang [deg]']} -- "
            + f"{p['datetime']:%Y-%m-%d %H:%M:%S}"
            if isinstance(p['elev_ang [deg]'], str)
            else
            f"{p['elev_ang [deg]']:{2}.{3}} deg. -- "
            + f"{p['datetime']:%Y-%m-%d %H:%M:%S}"
            for p in rscans_params]
    if nrows is None and ncols is None:
        if len(rscans_vars) <= 3:
            nrw = 1
            ncl = int(len(rscans_vars))
        elif len(rscans_vars) > 3 and len(rscans_vars) < 10 and len(rscans_vars) % 2:
            ncl = int(np.ceil(len(rscans_vars)/2))
            nrw = int(np.ceil(len(rscans_vars)/ncl))
        elif len(rscans_vars) >= 10 and len(rscans_vars) % 2:
            ncl = int(np.ceil(len(rscans_vars)/4))
            nrw = int(np.ceil(len(rscans_vars)/ncl))
        else:
            ncl = int(np.ceil(len(rscans_vars)/2))
            nrw = int(np.ceil(len(rscans_vars)/ncl))
    elif nrows is not None and ncols is None:
        if len(rscans_vars) <= 3:
            nrw = nrows
            ncl = int(len(rscans_vars))
        else:
            nrw = nrows
            ncl = int(np.ceil(len(rscans_vars)/nrw))
    elif ncols is not None and nrows is None:
        if len(rscans_vars) <= 3:
            nrw = 1
            ncl = ncols
        else:
            ncl = ncols
            nrw = int(np.ceil(len(rscans_vars)/ncl))
    else:
        nrw = nrows
        ncl = ncols
        if nrw * ncl < len(rscans_vars):
            print('Warning: Due to the selected grid, some variables may not '
                  + 'be displayed. Please adjust your settings to view all '
                  + 'available variables.')
    if coord_sys == 'rect' and cpy_features['status'] is False:
        if fig_size is None:
            fig_size = (15, 5)
        fig = plt.figure(figsize=fig_size)
        grgeor = [[i['grid_rectx'], i['grid_recty']] for i in rscans_georef]
        grid2 = ImageGrid(fig, 111, nrows_ncols=(nrw, ncl), label_mode="L",
                          cbar_location="right", cbar_mode="single",
                          cbar_size="10%", cbar_pad=0.25, axes_pad=(0.5, 0.75),
                          share_all=True)
        for ax, z, g, pr, pt in zip(grid2, [i[var2plot] for i in rscans_vars],
                                    grgeor, rscans_params, pttl):
            f1 = ax.pcolormesh(g[0], g[1], z, shading='auto', cmap=cmapp,
                               norm=normp)
            ax.set_title(f"{pt} \n {pr['site_name']} - PPI {var2plot}",
                         fontsize=12)
            ax.set_xlabel('Distance from the radar [km]', fontsize=12)
            ax.set_ylabel('Distance from the radar [km]', fontsize=12)
            ax.grid(True)
            ax.axes.set_aspect('equal')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
        if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
           or '[mm/h]' in var2plot):
            ax.cax.colorbar(f1, ticks=tcks, format=f'%.{cbtks_fmt}f')
        else:
            ax.cax.colorbar(f1)
        ax.cax.tick_params(direction='in', which='both', labelsize=12)
        ax.cax.set_title(var2plot[var2plot .find('['):], fontsize=12)
        # for ax, im_title in zip(grid2, ["(a)", "(b)", "(c)"]):
        #     t = add_inner_title(ax, im_title, loc='upper left')
        #     t.patch.set_ec("none")
        #     t.patch.set_alpha(0.5)
        if nrw*ncl > len(rscans_vars):
            for empax in range(nrw*ncl-len(rscans_vars)):
                grid2[-empax-1].remove()
        plt.tight_layout()
        plt.show()
    elif coord_sys == 'rect' and cpy_features['status']:
        if fig_size is None:
            fig_size = (16, 6)
        fig = plt.figure(figsize=fig_size)
        projection = ccrs.PlateCarree()
        axes_class = (GeoAxes, dict(map_projection=projection))
        grgeor = [[i[f'grid_{proj_suffix}x'], i[f'grid_{proj_suffix}y']]
                  for i in rscans_georef]
        grid2 = ImageGrid(fig, 111, nrows_ncols=(nrw, ncl), axes_pad=(.6, .9),
                          label_mode="L", cbar_mode="single", cbar_size="10%",
                          cbar_pad=0.75, cbar_location="right", share_all=True,
                          axes_class=axes_class)
        if data_proj:
            proj2 = data_proj
        else:
            raise TowerpyError('User must specify the projected coordinate'
                               ' system of the radar data e.g.'
                               ' ccrs.OSGB(approx=False) or ccrs.UTM(zone=32)')
        for ax1, z, g, pr, pt in zip(grid2, [i[var2plot] for i in rscans_vars],
                                     grgeor, rscans_params, pttl):
            ax1.set_title(f"{pt} \n {pr['site_name']} - PPI {var2plot}",
                          fontsize=12)
            if xlims and ylims:
                extx = xlims
                exty = ylims
                ax1.set_extent(extx+exty, crs=projection)
            if cpy_features['tiles']:
                if (cpy_features['tiles_source'] is None
                        or cpy_features['tiles_source'] == 'OSM'):
                    imtiles = cimgt.OSM()
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
                elif cpy_features['tiles_source'] == 'GoogleTiles':
                    if cpy_features['tiles_style'] is None:
                        imtiles = cimgt.GoogleTiles(style='street')
                        ax1.add_image(imtiles, cpy_features['tiles_res'],
                                      interpolation='spline36',
                                      alpha=cpy_features['alpha_tiles'])
                    else:
                        imtiles = cimgt.GoogleTiles(
                            style=cpy_features['tiles_style'])
                        ax1.add_image(imtiles, cpy_features['tiles_res'],
                                      # interpolation='spline36',
                                      alpha=cpy_features['alpha_tiles'])
                elif cpy_features['tiles_source'] == 'Stamen':
                    if cpy_features['tiles_style'] is None:
                        imtiles = cimgt.Stamen(style='toner')
                        ax1.add_image(imtiles, cpy_features['tiles_res'],
                                      interpolation='spline36',
                                      alpha=cpy_features['alpha_tiles'])
                    else:
                        imtiles = cimgt.Stamen(
                            style=cpy_features['tiles_style'])
                        ax1.add_image(imtiles, cpy_features['tiles_res'],
                                      interpolation='spline36',
                                      alpha=cpy_features['alpha_tiles'])
            if cpy_features['add_land']:
                ax1.add_feature(cfeature.LAND)
            if cpy_features['add_ocean']:
                ax1.add_feature(cfeature.OCEAN)
            if cpy_features['add_coastline']:
                ax1.add_feature(cfeature.COASTLINE)
            if cpy_features['add_borders']:
                ax1.add_feature(cfeature.BORDERS,
                                linestyle=cpy_features['borders_ls'])
            if cpy_features['add_lakes']:
                ax1.add_feature(cfeature.LAKES,
                                alpha=cpy_features['lakes_transparency'])
            if cpy_features['add_rivers']:
                ax1.add_feature(cfeature.RIVERS)
            if cpy_features['add_countries']:
                ax1.add_feature(states_provinces, edgecolor='black', ls=":")
            if cpy_features['add_provinces']:
                ax1.add_feature(countries, edgecolor='black')
            data_source = 'Natural Earth'
            data_license = 'public domain'
            # Add a text annotation for the license information to the
            # the bottom right corner.
            # text = AnchoredText(r'$\copyright$ {}; license: {}'
            #                     ''.format(SOURCE, LICENSE),
            #                     loc=4, prop={'size': 12}, frameon=True)
            # ax1.add_artist(text)
            print('\N{COPYRIGHT SIGN}'
                  + f'{data_source}; license: {data_license}')
            if cpy_features['tiles_source'] == 'Stamen':
                print('\N{COPYRIGHT SIGN}' + 'Map tiles by Stamen Design, '
                      + 'under CC BY 3.0. Data by OpenStreetMap, under ODbL.')
            gl = ax1.gridlines(draw_labels=True, dms=False,
                               x_inline=False, y_inline=False)
            gl.xlabel_style = {'size': 11}
            gl.ylabel_style = {'size': 11}
            gl.top_labels = False
            gl.right_labels = False
            # ax1.set_title(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=14)
            # lon_formatter = LongitudeFormatter(number_format='.4f',
            #                                 degree_symbol='',
            #                                dateline_direction_label=True)
            # lat_formatter = LatitudeFormatter(number_format='.0f',
            #                                    degree_symbol=''
            #                                   )
            # ax1.xaxis.set_major_formatter(lon_formatter)
            # ax1.yaxis.set_major_formatter(lat_formatter)
            # plotunits = [i[i.find('['):]
            #              for i in rad_vars.keys() if var2plot == i][0]
            mappable = ax1.pcolormesh(g[0], g[1], z, transform=proj2,
                                      shading='auto', cmap=cmapp, norm=normp,
                                      alpha=cpy_features['alpha_rad'])
            if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
               or '[mm/h]' in var2plot):
                # ticks = bnd.get(var2plot[var2plot.find('['):])
                if len(tcks) > 20:
                    tcks = tcks[::5]
                grid2.cbar_axes[0].colorbar(mappable, ticks=tcks,
                                            format=f'%.{cbtks_fmt}f')
            else:
                grid2.cbar_axes[0].colorbar(mappable)
            ax1.cax.set_title(var2plot[var2plot .find('['):], fontsize=12)
            # ax1.axes.set_aspect('equal')
        if nrw*ncl > len(rscans_vars):
            for empax in range(nrw*ncl-len(rscans_vars)):
                grid2[-empax-1].remove()
        plt.tight_layout()
        plt.show()


def plot_cone_coverage(rad_georef, rad_params, rad_vars, var2plot=None,
                       vars_bounds=None, xlims=None, ylims=None, zlims=[0, 8],
                       limh=8, ucmap=None, unorm=None, cbticks=None,
                       cb_ext=None, fig_size=None):
    """
    Display a 3-D representation of the radar cone coverage.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars : dict
        Dict containing radar variables to plot.
    var2plot : str, optional
        Key of the radar variable to plot. The default is None. This option
        will plot ZH or the 'first' element in the rad_vars dict.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    zlims : 2-element tuple or list, optional
        Set the z-axis view limits [min, max]. The default is None.
    limh : int or float, optional
        Set a height limit to the plot. The default is None.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    cbticks : dict, optional
        Modifies the default ticks' location (dict values) and labels
        (dict keys) in the colour bar.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    from matplotlib.colors import LightSource

    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot, rad_vars.keys(), vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    #
    if var2plot is None:
        var2plot = v2p
    #
    cmapp = cmaph.get(var2plot[var2plot.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
    # dtdes0 = f"[{rad_params['site_name']}]"
    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} deg. -- "
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # txtboxc = (0, -.09)
    # fc, ec = 'w', 'k'

    limidx = [rut.find_nearest(row, limh)
              for row in rad_georef['beam_height [km]']]

    m = np.ma.masked_invalid(rad_vars[var2plot]).mask
    for n, rows in enumerate(m):
        rows[limidx[n]:] = 1
    R = rad_vars[var2plot]

    X, Y = rad_georef['grid_rectx'], rad_georef['grid_recty']
    Z = np.resize(rad_georef['beam_height [km]'], R.shape)
    Z = np.resize(rad_georef['beam_height [km]'], R.shape)

    X = np.ma.array(X, mask=m)
    Y = np.ma.array(Y, mask=m)
    Z = np.ma.array(Z, mask=m)
    R = np.ma.array(R, mask=m)

    ls = LightSource(0, 0)

    rgb = ls.shade(R, cmap=cmapp, norm=normp, vert_exag=0.1, blend_mode='soft')
    if fig_size is None:
        fig_size = (12, 8)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=fig_size)

    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap=cmapp, norm=normp, facecolors=rgb,
                    rstride=1, cstride=8,
                    # rcount=360, ccount=600,
                    # rcount=360, ccount=150,
                    linewidth=0, antialiased=True, shade=False,)
    if cbticks is not None:
        mappable2 = ax.contourf(X, Y, R, zdir='z', offset=0, levels=tcks,
                                cmap=cmapp, norm=normp, antialiased=True)
    else:
        mappable2 = ax.contourf(X, Y, R, zdir='z', offset=0,
                                levels=normp.boundaries,
                                cmap=cmapp, norm=normp, extend=normp.extend,
                                antialiased=True)
    # Customize the axis.
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    ax.view_init(elev=10)
    ax.tick_params(axis='both', labelsize=14)

    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    if cbticks is not None:
        cb1 = fig.colorbar(mappable2, shrink=0.4, aspect=5, norm=normp,
                           cmap=cmapp, ticks=list(cbticks.values()),
                           format=mticker.FixedFormatter(list(cbticks.keys())))
    else:
        if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
           or '[mm/h]' in var2plot):
            cb1 = fig.colorbar(
                mappable2, shrink=0.4, aspect=5, norm=normp, cmap=cmapp,
                ticks=tcks, format=f'%.{cbtks_fmt}f')
        else:
            cb1 = fig.colorbar(
                mappable2, shrink=0.4, aspect=5, norm=normp, cmap=cmapp)
    cb1.ax.tick_params(direction='in', axis='both', labelsize=14)
    cb1.ax.set_title(var2plot[var2plot .find('['):], fontsize=14)

    ax.set_title(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=16)
    ax.set_xlabel('Distance from the radar [km]', fontsize=14, labelpad=15)
    ax.set_ylabel('Distance from the radar [km]', fontsize=14, labelpad=15)
    ax.set_zlabel('Height [km]', fontsize=14, labelpad=15)
    plt.tight_layout()
    plt.show()


def plot_snr(rad_georef, rad_params, snr_data, min_snr, coord_sys='rect',
             ucmap_snr=None, fig_size=None):
    """
    Display the results of the SNR classification.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    snr_data : dict
        Results of the SNR_Classif method.
    proj : 'rect' or 'polar', optional
        Coordinates system (polar or rectangular). The default is 'rect'.
    """
    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} deg. -- "
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2
    if fig_size is None:
        fig_size = (10.5, 6.5)
    if coord_sys == 'polar':
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=fig_size,
                                       subplot_kw=dict(projection='polar'))

        f2 = ax2.pcolormesh(rad_georef['theta'], rad_georef['rho'],
                            np.flipud(snr_data['snr [dB]']), shading='auto',
                            cmap='tpylsc_rad_ref')
        # ax2.axes.set_aspect('equal')
        ax2.grid(color='gray', linestyle=':')
        ax2.set_theta_zero_location('N')
        ax2.set_thetagrids(np.arange(0, 360, 90))
        # ax2.set_yticklabels([])
        cb2 = plt.colorbar(f2, ax=ax2, extend='both', orientation='horizontal',
                           # shrink=0.5,
                           )
        cb2.ax.tick_params(direction='in', axis='both', labelsize=14)
        cb2.ax.set_title('SNR [dB]', fontsize=14, y=-2.5)

        ax3.set_title(f'Signal (SNR > minSNR [{min_snr:.2f}])', fontsize=14,
                      y=-0.15)
        ax3.pcolormesh(rad_georef['theta'], rad_georef['rho'],
                       np.flipud(snr_data['snrclass']), shading='auto',
                       cmap=mpl.colormaps['tpylc_div_yw_gy_bu'])
        # ax3.axes.set_aspect('equal')
        ax3.grid(color='w', linestyle=':')
        ax3.set_theta_zero_location('N')
        ax3.set_thetagrids(np.arange(0, 360, 90))
        # ax3.set_yticklabels([])
        mpl.colormaps['tpylc_div_yw_gy_bu'].set_bad(color='#505050')
        plt.show()

    elif coord_sys == 'rect':
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=fig_size,
                                       sharex=True, sharey=True)
        fig.suptitle(f'{ptitle}', fontsize=16)
        # Plots the SNR
        f2 = ax2.pcolormesh(rad_georef['grid_rectx'], rad_georef['grid_recty'],
                            snr_data['snr [dB]'], shading='auto',
                            cmap='tpylsc_rad_ref')
        ax2_divider = make_axes_locatable(ax2)
        cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
        cb2 = fig.colorbar(f2, cax=cax2, extend='max',
                           orientation='horizontal')
        cb2.ax.tick_params(direction='in', labelsize=10)
        # cb2.ax.set_xticklabels(cb2.ax.get_xticklabels(), rotation=90)
        cb2.ax.set_title('SNR [dB]', fontsize=14)
        # cb2.ax.set_ylabel('[dB]', fontsize=12, labelpad=0)
        cax2.xaxis.set_ticks_position("top")
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.set_ylabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax2.set_xlabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax2.axes.set_aspect('equal')
        # Plots the Signal detection
        f3 = ax3.pcolormesh(rad_georef['grid_rectx'], rad_georef['grid_recty'],
                            snr_data['snrclass'],
                            cmap=mpl.colormaps['tpylc_div_yw_gy_bu'],
                            vmin=0, vmax=6,
                            )
        ax3_divider = make_axes_locatable(ax3)
        cax3 = ax3_divider.append_axes("top", size="7%", pad="2%")
        cb3 = fig.colorbar(f3, cax=cax3, orientation='horizontal')
        cb3.ax.tick_params(direction='in', labelsize=10)
        cb3.ax.set_title(f'Signal detection - SNR >= minSNR [{min_snr:.2f}]',
                         fontsize=14)
        cax3.xaxis.set_ticks_position("top")
        cb3.set_ticks(ticks=[1., 3., 6.],
                      labels=['Signal', 'Noise', ''])
        ax3.set_xlabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax3.tick_params(axis='both', which='major', labelsize=10)
        ax3.axes.set_aspect('equal')
        plt.tight_layout()
        plt.show()


def plot_nmeclassif(rad_georef, rad_params, nme_classif, echoesID,
                    clutter_map=None, xlims=None, ylims=None, fig_size=None):
    """
    Plot a set of PPIs of polarimetric variables.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    nme_classif : dict
        Results of the NME_ID method.
    clutter_map : array, optional
        Clutter map used for the NME_ID method. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    """
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # fc, ec = 'w', 'k'
    # =========================================================================
    # Plot the Clutter classification
    # =========================================================================
    if fig_size is None:
        fig_size = (6, 6.15)
    clcdummy = nme_classif[nme_classif == echoesID['clutter']]
    if not clcdummy.size:
        nme_classif[0, 0] = echoesID['clutter']
    plot_ppi(rad_georef, rad_params, {'classif [EC]': nme_classif},
             cbticks=echoesID, ucmap='tpylc_div_yw_gy_bu')
    plt.tight_layout()
    # # txtboxc = (0, -.09)
    # # ax.annotate('| Created using Towerpy |', xy=txtboxc, fontsize=8,
    # #             xycoords='axes fraction', va='center', ha='center',
    # #             bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
    # =========================================================================
    # Plot the Clutter Map
    # =========================================================================
    if clutter_map is not None:
        norm = mpc.BoundaryNorm(boundaries=np.linspace(0, 100, 11),
                                ncolors=256)
        plot_ppi(rad_georef, rad_params,
                 {'Clutter probability [%]': clutter_map*100},
                 unorm={'Clutter probability [%]': norm},
                 ucmap='tpylsc_useq_bupkyw')
        # txtboxc = (0, -.09)
        # ax.annotate('| Created using Towerpy |', xy=txtboxc, fontsize=8,
        #             xycoords='axes fraction', va='center', ha='center',
        #             bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
        plt.tight_layout()
    plt.show()


def plot_zhattcorr(rad_georef, rad_params, rad_vars_att, rad_vars_attcorr,
                   vars_bounds=None, mlyr=None, xlims=None, ylims=None,
                   fig_size1=None, fig_size2=None):
    """
    Plot the results of the ZH attenuation correction method.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars_att : dict
        Radar variables not corrected for attenuation.
    rad_vars_attcorr : dict
        Results of the AttenuationCorection method.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped. The default are:
            {'ZH [dBZ]': [-10, 60, 15],
             'ZDR [dB]': [-2, 6, 17],
             'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
             'rhoHV [-]': [0.3, .9, 1],
             'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
             'LDR [dB]': [-35, 0, 11],
             'Rainfall [mm/h]': [0.1, 64, 11]}
    mlyr : MeltingLayer Class, optional
        Plot the melting layer height. ml_top (float, int, list or np.array)
        and ml_bottom (float, int, list or np.array) must be explicitly
        defined. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    """
    lpv = {'ZH [dBZ]': [-10, 60, 15], 'PhiDP [deg]': [0, 180, 10],
           'KDP [deg/km]': [-2, 6, 17], 'AH [dB/km]': [0, .1, 11],
           'alpha [-]': [0, 0.2, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)
# =============================================================================
    bnd = {key[key.find('['):]: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
# =============================================================================
    dnorm = {key: mpc.BoundaryNorm(
        value, mpl.colormaps['tpylsc_rad_pvars'].N, extend='both')
             for key, value in bnd.items()}
    if '[dBZ]' in bnd.keys():
        dnorm['[dBZ]'] = mpc.BoundaryNorm(
            bnd['[dBZ]'], mpl.colormaps['tpylsc_rad_ref'].N, extend='both')
    if '[dB]' in bnd.keys():
        dnorm['[dB]'] = mpc.BoundaryNorm(
            bnd['[dB]'], mpl.colormaps['tpylsc_rad_2slope'].N, extend='both')
    if '[dB/km]' in bnd.keys():
        dnorm['[dB/km]'] = mpc.BoundaryNorm(
            bnd['[dB/km]'], mpl.colormaps['tpylsc_rad_pvars'].N,
            extend='max')
    if '[-]' in bnd.keys():
        dnorm['[-]'] = mpc.BoundaryNorm(
            bnd['[-]'], mpl.colormaps['tpylsc_useq_fiery'].N,
            extend='neither')
# =============================================================================
    if mlyr is not None:
        if isinstance(mlyr.ml_top, (int, float)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_top, (np.ndarray, list, tuple)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        if isinstance(mlyr.ml_bottom, (int, float)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_bottom, (np.ndarray, list, tuple)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        mlt_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlt_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlb_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])
        mlb_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])

    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} deg. -- "
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2

    # =========================================================================
    # Creates plots for ZH attenuation correction results.
    # =========================================================================
    mosaic = 'ABC'
    if fig_size1 is None:
        fig_size1 = (16, 5)
    if fig_size2 is None:
        fig_size2 = (6, 5)

    fig_mos1 = plt.figure(figsize=fig_size1, constrained_layout=True)
    ax_idx = fig_mos1.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_vars_att.items():
        if '[dBZ]' in key:
            cmap = mpl.colormaps['tpylsc_rad_ref']
            # norm = dnorm.get('n'+key)
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['A'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['A'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx['A'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['A'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['A'].legend(loc='upper left')
    if xlims is not None:
        ax_idx['A'].set_xlim(xlims)
    if ylims is not None:
        ax_idx['A'].set_ylim(ylims)
    plt.colorbar(fzhna, ax=ax_idx['A']).ax.tick_params(labelsize=10)
    ax_idx['A'].grid(True)
    ax_idx['A'].axes.set_aspect('equal')
    ax_idx['A'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if '[dBZ]' in key:
            cmap = mpl.colormaps['tpylsc_rad_ref']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['B'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['B'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx['B'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['B'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['B'].legend(loc='upper left')
    if xlims is not None:
        ax_idx['B'].set_xlim(xlims)
    if ylims is not None:
        ax_idx['B'].set_ylim(ylims)
    plt.colorbar(fzhna, ax=ax_idx['B']).ax.tick_params(labelsize=10)
    ax_idx['B'].grid(True)
    ax_idx['B'].axes.set_aspect('equal')
    ax_idx['B'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if 'AH' in key:
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['C'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm
                                           )
            ax_idx['C'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['C'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['C'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['C'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['C']).ax.tick_params(labelsize=10)
    ax_idx['C'].grid(True)
    ax_idx['C'].axes.set_aspect('equal')
    ax_idx['C'].tick_params(axis='both', labelsize=10)

    # =========================================================================
    # Creates plots for PHIDP attenuation correction results.
    # =========================================================================
    fig_mos2 = plt.figure(figsize=fig_size1, constrained_layout=True)
    ax_idx2 = fig_mos2.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_vars_att.items():
        if '[deg]' in key:
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx2['A'].pcolormesh(rad_georef['grid_rectx'],
                                            rad_georef['grid_recty'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['A'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx2['A'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx2['A'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx2['A'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx2['A']).ax.tick_params(labelsize=10)
    ax_idx2['A'].grid(True)
    ax_idx2['A'].axes.set_aspect('equal')
    ax_idx2['A'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if key == 'PhiDP [deg]':
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx2['B'].pcolormesh(rad_georef['grid_rectx'],
                                            rad_georef['grid_recty'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['B'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx2['B'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx2['B'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx2['B'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx2['B']).ax.tick_params(labelsize=10)
    ax_idx2['B'].grid(True)
    ax_idx2['B'].axes.set_aspect('equal')
    ax_idx2['B'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if key == 'PhiDP* [deg]':
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            # norm = dnorm.get('n'+key.replace('*', ''))
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx2['C'].pcolormesh(rad_georef['grid_rectx'],
                                            rad_georef['grid_recty'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['C'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx2['C'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx2['C'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx2['C'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx2['C']).ax.tick_params(labelsize=10)
    ax_idx2['C'].grid(True)
    ax_idx2['C'].axes.set_aspect('equal')
    ax_idx2['C'].tick_params(axis='both', labelsize=10)

    # =========================================================================
    # Creates plots for attenuation correction vars.
    # =========================================================================
    fig_mos3, ax_idx3 = plt.subplots(figsize=fig_size2)
    for key, value in rad_vars_attcorr.items():
        if key == 'alpha [-]':
            # cmap = 'tpylsc_rad_pvars'
            cmap = 'tpylsc_useq_fiery'
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx3.pcolormesh(rad_georef['grid_rectx'],
                                       rad_georef['grid_recty'], value,
                                       shading='auto', cmap=cmap,
                                       norm=norm
                                       )
            ax_idx3.set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx3.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx3.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx3.legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx3).ax.tick_params(labelsize=10)
    ax_idx3.grid(True)
    ax_idx3.axes.set_aspect('equal')
    ax_idx3.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    fig_mos4, ax_idx4 = plt.subplots(figsize=fig_size2)
    for key, value in rad_vars_attcorr.items():
        if 'PIA' in key:
            # cmap = 'tpylsc_rad_pvars'
            cmap = 'tpylsc_useq_fiery'
            fzhna = ax_idx4.pcolormesh(rad_georef['grid_rectx'],
                                       rad_georef['grid_recty'], value,
                                       shading='auto', cmap=cmap,
                                       # norm=norm
                                       )
            ax_idx4.set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx4.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx4.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx4.legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx4).ax.tick_params(labelsize=10)
    ax_idx4.grid(True)
    ax_idx4.axes.set_aspect('equal')
    ax_idx4.tick_params(axis='both', labelsize=10)
    plt.tight_layout()


def plot_zdrattcorr(rad_georef, rad_params, rad_vars_att, rad_vars_attcorr,
                    vars_bounds=None, mlyr=None, xlims=None, ylims=None,
                    fig_size1=None, fig_size2=None):
    """
    Plot the results of the ZDR attenuation correction method.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars_att : dict
        Radar variables not corrected for attenuation.
    rad_vars_attcorr : dict
        Results of the AttenuationCorection method.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped. The default are:
            {'ZH [dBZ]': [-10, 60, 15],
             'ZDR [dB]': [-2, 6, 17],
             'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
             'rhoHV [-]': [0.3, .9, 1],
             'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
             'LDR [dB]': [-35, 0, 11],
             'Rainfall [mm/h]': [0.1, 64, 11]}
    mlyr : MeltingLayer Class, optional
        Plot the melting layer height. ml_top (float, int, list or np.array)
        and ml_bottom (float, int, list or np.array) must be explicitly
        defined. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    """
    lpv = {'ZDR [dB]': [-2, 6, 17], 'ADP [dB/km]': [0, 2.5, 20],
           'beta [-]': [0, 0.1, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)
# =============================================================================
    bnd = {key[key.find('['):]: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
# =============================================================================
    dnorm = {key: mpc.BoundaryNorm(
        value, mpl.colormaps['tpylsc_rad_pvars'].N, extend='both')
             for key, value in bnd.items()}
    if '[dBZ]' in bnd.keys():
        dnorm['[dBZ]'] = mpc.BoundaryNorm(
            bnd['[dBZ]'], mpl.colormaps['tpylsc_rad_ref'].N, extend='both')
    if '[dB]' in bnd.keys():
        dnorm['[dB]'] = mpc.BoundaryNorm(
            bnd['[dB]'], mpl.colormaps['tpylsc_rad_2slope'].N, extend='both')
    if '[dB/km]' in bnd.keys():
        dnorm['[dB/km]'] = mpc.BoundaryNorm(
            bnd['[dB/km]'], mpl.colormaps['tpylsc_rad_pvars'].N,
            extend='max')
    if '[-]' in bnd.keys():
        dnorm['[-]'] = mpc.BoundaryNorm(
            bnd['[-]'], mpl.colormaps['tpylsc_useq_fiery'].N,
            extend='max')
# =============================================================================
    if mlyr is not None:
        if isinstance(mlyr.ml_top, (int, float)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_top, (np.ndarray, list, tuple)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        if isinstance(mlyr.ml_bottom, (int, float)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_bottom, (np.ndarray, list, tuple)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        mlt_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlt_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlb_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])
        mlb_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])

    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} deg. -- "
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2

    # =========================================================================
    # Creates plots for ZDR attenuation correction results.
    # =========================================================================
    mosaic = 'DEF'
    if fig_size1 is None:
        fig_size1 = (16, 5)
    if fig_size2 is None:
        fig_size2 = (6, 5)

    fig_mos1 = plt.figure(figsize=fig_size1, constrained_layout=True)
    ax_idx = fig_mos1.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_vars_att.items():
        if '[dB]' in key:
            cmap = mpl.colormaps['tpylsc_rad_2slope']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['D'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['D'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx['D'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['D'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['D'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['D']).ax.tick_params(labelsize=10)
    ax_idx['D'].grid(True)
    ax_idx['D'].axes.set_aspect('equal')
    ax_idx['D'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if '[dB]' in key:
            cmap = mpl.colormaps['tpylsc_rad_2slope']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['E'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['E'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx['E'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['E'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['E'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['E']).ax.tick_params(labelsize=10)
    ax_idx['E'].grid(True)
    ax_idx['E'].axes.set_aspect('equal')
    ax_idx['E'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if 'ADP' in key:
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['F'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['F'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['F'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['F'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['F'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['F']).ax.tick_params(labelsize=10)
    ax_idx['F'].grid(True)
    ax_idx['F'].axes.set_aspect('equal')
    ax_idx['F'].tick_params(axis='both', labelsize=10)

    # =========================================================================
    # Creates plots for attenuation correction vars.
    # =========================================================================
    fig_mos3, ax_idx3 = plt.subplots(figsize=fig_size2)
    for key, value in rad_vars_attcorr.items():
        if key == 'beta [-]':
            cmap = 'tpylsc_useq_fiery'
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx3.pcolormesh(rad_georef['grid_rectx'],
                                       rad_georef['grid_recty'], value,
                                       shading='auto', cmap=cmap, norm=norm)
            ax_idx3.set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx3.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx3.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx3.legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx3).ax.tick_params(labelsize=10)
    ax_idx3.grid(True)
    ax_idx3.axes.set_aspect('equal')
    ax_idx3.tick_params(axis='both', labelsize=10)
    plt.tight_layout()


def plot_radprofiles(rad_profs, beam_height, mlyr=None, stats=None, ylims=None,
                     vars_bounds=None, colours=False, unorm=None, ucmap=None,
                     cb_ext=None, fig_size=None):
    """
    Display a set of profiles of polarimetric variables.

    Parameters
    ----------
    rad_profs : dict
        Profiles generated by the PolarimetricProfiles class.
    beam_height : array
        The beam height.
    mlyr : MeltingLayer Class, optional
        Plots the melting layer within the polarimetric profiles.
        The default is None.
    stats : dict, optional
        Statistics of the profiles generation computed by the
        PolarimetricProfiles class. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max] between which radar variables are
        to be plotted.
    colours : Bool, optional
        Creates coloured profiles using norm to map colormaps.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    fontsizelabels = 20
    fontsizetitle = 25
    fontsizetick = 18
    prftype = getattr(rad_profs, 'profs_type').lower()

    # ttxt_elev = f"{rad_profs.elev_angle:{2}.{3}} Deg."
    # ttxt_dt = f"{rad_profs.scandatetime:%Y-%m-%d %H:%M:%S}"
    # ttxt = dtdes1+ttxt_dt
    if isinstance(rad_profs.elev_angle, str):
        dtdes1 = f"{rad_profs.elev_angle} -- "
    else:
        dtdes1 = f"{rad_profs.elev_angle:{2}.{3}} deg. -- "
    dtdes2 = f"{rad_profs.scandatetime:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2

    if fig_size is None:
        fig_size = (14, 10)

    def make_colorbar(ax1, mappable, **kwargs):
        ax1_divider = make_axes_locatable(ax1)
        orientation = kwargs.pop('orientation', 'vertical')
        if orientation == 'vertical':
            loc = 'right'
        elif orientation == 'horizontal':
            loc = 'top'
        cax = ax1_divider.append_axes(loc, '7%', pad='2.5%',
                                      axes_class=plt.Axes)
        ax1.get_figure().colorbar(mappable, cax=cax,
                                  orientation=orientation,
                                  ticks=ticks,
                                  format=f'%.{cbtks_fmt}f')
        cax.tick_params(direction='in', labelsize=10, rotation=90)
        cax.xaxis.set_ticks_position('top')
    if rad_profs.profs_type.lower() == 'vps':
        rprofs = rad_profs.vps
        fig, ax = plt.subplots(1, len(rad_profs.vps), figsize=fig_size,
                               sharey=True)
        fig.suptitle(f'Vertical profiles of polarimetric variables'
                     '\n' f'{ptitle}',
                     fontsize=fontsizetitle)
    elif rad_profs.profs_type.lower() == 'qvps':
        rprofs = rad_profs.qvps
        fig, ax = plt.subplots(1, len(rad_profs.qvps), figsize=fig_size,
                               sharey=True)
        fig.suptitle('Quasi-Vertical profiles of polarimetric variables \n'
                     f'{ptitle}',
                     fontsize=fontsizetitle)
    for n, (a, (key, value)) in enumerate(zip(ax.flatten(), rprofs.items())):
        lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, ticks = pltparams(
            key, getattr(rad_profs, prftype).keys(), vars_bounds, ucmap=ucmap,
            unorm=unorm, cb_ext=cb_ext)
        if key == 'rhoHV [-]':
            ticks = ticks
        else:
            ticks = None
        if colours is False:
            a.plot(value, beam_height, 'k')
        elif colours:
            if unorm is not None:
                dnorm.update(unorm)
            cmapp = cmaph.get(key[key.find('['):],
                              mpl.colormaps['tpylsc_rad_pvars'])
            points = np.array([value, beam_height]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmapp,
                                norm=dnorm.get(key[key.find('['):]))
            # Set the values used for colormapping
            lc.set_array(value)
            lc.set_linewidth(2)
            # line = a.add_collection(lc)
            a.add_collection(lc)
            make_colorbar(a, lc, orientation='horizontal')
            if np.isfinite(np.nanmin(value)) and np.isfinite(np.nanmax(value)):
                a.set_xlim([np.nanmin(value), np.nanmax(value)])
        # if stats:
        #     a.fill_betweenx(beam_height,
        #                     value + stats.get(key, value*np.nan),
        #                     value - stats.get(key, value*np.nan),
        #                     alpha=0.4, color='gray', label='std')
        if stats == 'std_dev' or stats == 'sem':
            if rad_profs.profs_type.lower() == 'vps':
                a.fill_betweenx(beam_height,
                                value + rad_profs.vps_stats[stats][key],
                                value - rad_profs.vps_stats[stats][key],
                                alpha=0.4, label=f'{stats}')
            if rad_profs.profs_type.lower() == 'qvps':
                a.fill_betweenx(beam_height,
                                value + rad_profs.qvps_stats[stats][key],
                                value - rad_profs.qvps_stats[stats][key],
                                alpha=0.4, label=f'{stats}')
            # a.fill_betweenx(beam_height,
            #                 value + stats.get(key, value*np.nan),
            #                 value - stats.get(key, value*np.nan),
            #                 alpha=0.4, color='gray', label='std')
        if n == 0:
            a.set_ylabel('Height [km]', fontsize=fontsizelabels, labelpad=15)
        a.tick_params(axis='both', labelsize=fontsizetick)
        a.grid(True)
        if vars_bounds:
            if key in lpv:
                if key == 'rhoHV [-]':
                    a.set_xlim(lpv.get(key)[0], lpv.get(key)[2])
                else:
                    a.set_xlim(lpv.get(key)[:2])
        if mlyr:
            a.axhline(mlyr.ml_top, c='tab:blue', ls='dashed', lw=5,
                      alpha=.5, label='$ML_{top}$')
            a.axhline(mlyr.ml_bottom, c='tab:purple', ls='dashed', lw=5,
                      alpha=.5, label='$ML_{bottom}$')
            a.legend(loc='upper right', fontsize=fontsizetick)
        if key == 'ZH [dBZ]':
            a.set_xlabel('$Z_{H}$ [dBZ]', fontsize=fontsizelabels)
        elif key == 'ZDR [dB]':
            a.set_xlabel('$Z_{DR}$ [dB]', fontsize=fontsizelabels)
        elif key == 'rhoHV [-]':
            a.set_xlabel(r'$ \rho_{HV}$ [-]', fontsize=fontsizelabels)
        elif key == 'PhiDP [deg]':
            a.set_xlabel(r'$ \Phi_{DP}$ [deg]', fontsize=fontsizelabels)
        elif key == 'V [m/s]':
            a.set_xlabel('V [m/s]', fontsize=fontsizelabels)
        elif key == 'gradV [dV/dh]' and rad_profs.profs_type.lower() == 'vps':
            a.set_xlabel('grad V [dV/dh]', fontsize=fontsizelabels)
        elif key == 'KDP [deg/km]':
            a.set_xlabel('$K_{DP}$'+r'$\left [\frac{deg}{km}\right ]$',
                         fontsize=fontsizelabels)
        else:
            a.set_xlabel(key, fontsize=fontsizelabels)
        if ylims:
            a.set_ylim(ylims)
        else:
            a.set_ylim(0, 10)
    plt.show()
    plt.tight_layout()


def plot_rdqvps(rscans_georef, rscans_params, tp_rdqvp, spec_range=None,
                mlyr=None, ylims=None, vars_bounds=None, ucmap=None,
                cb_ext=None, all_desc=False, fig_size=None):
    """
    Display a set of RD-QVPS of polarimetric variables.

    Parameters
    ----------
    rscans_georef : List
        List of georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others, corresponding to each QVP.
    rscans_params : List
        List of radar technical details corresponding to each QVP.
    tp_rdqvp : PolarimetricProfiles Class
        Outputs of the RD-QVPs function.
    spec_range : int, optional
        Range from the radar within which the RD-QVPS were built.
    mlyr : MeltingLayer Class, optional
        Plots the melting layer within the polarimetric profiles.
        The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    vars_bounds : dict containing key and 2-element tuple or list, optional
        Boundaries [min, max] between which radar variables are
        to be plotted.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    all_desc : bool, optional
        If True, plots the initial QVPs used to compute the RD-QPVs.
        The default is True.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    if fig_size is None:
        fig_size = (14, 10)

    fontsizelabels = 20
    fontsizetitle = 25
    fontsizetick = 18
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        None, tp_rdqvp.rd_qvps.keys(), vars_bounds, ucmap=ucmap, cb_ext=cb_ext)
    if vars_bounds:
        lpv.update(vars_bounds)
    cmaph = mpl.colormaps['Spectral'](
        np.linspace(0, 1, len(rscans_params)))
    if ucmap is not None:
        if isinstance(ucmap, str):
            cmaph = mpl.colormaps[ucmap](np.linspace(0, 1, len(rscans_params)))
        else:
            cmaph = ucmap(np.linspace(0, 1, len(rscans_params)))
    # ttxt = f"{rscans_params[0]['datetime']:%Y-%m-%d %H:%M:%S}"
    dt1 = min([i['datetime'] for i in rscans_params])
    dt2 = max([i['datetime'] for i in rscans_params])
    ttxt = (f"{dt1:%Y-%m-%d %H:%M:%S} - {dt2:%H:%M:%S}")

    mosaic = [chr(ord('@')+c+1) for c in range(len(tp_rdqvp.rd_qvps)+1)]
    mosaic = f'{"".join(mosaic)}'

    fig = plt.figure(layout="constrained", figsize=fig_size)
    fig.suptitle('RD-QVPs of polarimetric variables \n' f'{ttxt}',
                 fontsize=fontsizetitle)
    axd = fig.subplot_mosaic(mosaic, sharey=True, height_ratios=[5])

    if all_desc:
        for c, i in enumerate(tp_rdqvp.qvps_itp):
            for n, (a, (key, value)) in enumerate(zip(axd, i.items())):
                axd[a].plot(value, tp_rdqvp.georef['profiles_height [km]'],
                            color=cmaph[c], ls='--',
                            label=(f"{rscans_params[c]['elev_ang [deg]']:.1f}"
                                   + r"$^{\circ}$"))
                # axd[a].legend(loc='upper right')
    if not all_desc:
        i = tp_rdqvp.rd_qvps
    for n, (a, (key, value)) in enumerate(zip(axd, i.items())):
        axd[a].plot(tp_rdqvp.rd_qvps[key],
                    tp_rdqvp.georef['profiles_height [km]'], 'k', lw=3,
                    label='RD-QVP')
        axd[a].legend(loc='upper right')
        if vars_bounds:
            if key in lpv:
                axd[a].set_xlim(lpv.get(key))
            else:
                axd[a].set_xlim([np.nanmin(value), np.nanmax(value)])
        if mlyr:
            axd[a].axhline(mlyr.ml_top, c='tab:blue', ls='dashed', lw=5,
                           alpha=.5, label='$ML_{top}$')
            axd[a].axhline(mlyr.ml_bottom, c='tab:purple', ls='dashed', lw=5,
                           alpha=.5, label='$ML_{bottom}$')
        if ylims:
            axd[a].set_ylim(ylims)
        axd[a].set_xlabel(f'{key}', fontsize=fontsizelabels)
        if n == 0:
            axd[a].set_ylabel('Height [km]', fontsize=fontsizelabels,
                              labelpad=10)
        axd[a].tick_params(axis='both', labelsize=fontsizetick)
        axd[a].grid(True)

    scan_st = axd[mosaic[-1]]
    for c, i in enumerate(rscans_georef):
        scan_st.plot(i['range [m]']/1000, i['beam_height [km]'][0],
                     color=cmaph[c], ls='--',
                     label=(f"{rscans_params[c]['elev_ang [deg]']:.1f}"
                            + r"$^{\circ}$"))
        # scan_st.plot(i['range [m]']/-1000, i['beam_height [km]'][0],
        #               color=cmaph[c], ls='--')
    if spec_range:
        scan_st.axvline(spec_range, c='k', lw=3, label=f'RD={spec_range}')
    scan_st.set_xlabel('Range [km]', fontsize=fontsizelabels)
    scan_st.tick_params(axis='both', labelsize=fontsizetick)
    scan_st.grid(True)
    scan_st.legend(loc='upper right')


def plot_offsetcorrection(rad_georef, rad_params, rad_var, var_offset=0,
                          fig_size=None, var_name='PhiDP [deg]',
                          cmap='tpylsc_div_lbu_w_rd'):
    """
    Plot the offset detection method from ZDR/PhiDP_Calibration Class.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_var : dict
        PPI scan of the radar variable used to detect the offset.
    var_name : str
        Key of the radar variable used to detect the offset.
    cmap : colormap, optional
        User-defined colormap. The default is 'tpylsc_div_lbu_w_rd'.
    """
    var_mean = np.array([np.nanmean(i) for i in rad_var])
    # var_mean = np.zeros_like(rad_georef['azim [rad]']) + var_offset
    if var_name == 'PhiDP [deg]':
        label1 = r'$\Phi_{DP}$'
        labelm = r'$\overline{\Phi_{DP}}$'
        labelo = r'$\Phi_{DP}$ offset'
        dval = 3
        dof = 10
    elif var_name == 'ZDR [dB]':
        label1 = '$Z_{DR}$'
        labelm = r'$\overline{Z_{DR}}$'
        labelo = r'$Z_{DR}$ offset'
        dval = 0.1
        dof = 1
    if fig_size is None:
        fig_size = (8, 8)

    fig, ax = plt.subplots(figsize=fig_size,
                           subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} deg. -- "
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2
    ax.set_title(ptitle, fontsize=16)
    ax.grid(color='gray', linestyle=':')
    ax.set_theta_zero_location('N', offset=0)
    # =========================================================================
    # Plot the radar variable values at each azimuth
    # =========================================================================
    ax.scatter((np.ones_like(rad_var.T) * [rad_georef['azim [rad]']]).T,
               rad_var, s=5, c=rad_var, cmap=cmap, label=label1,
               norm=mpc.SymLogNorm(linthresh=.01, linscale=.01, base=2,
                                   vmin=var_mean.mean()-dval,
                                   vmax=var_mean.mean()+dval))
    # =========================================================================
    # Plot the radar variable mean value of each azimuth
    # =========================================================================
    ax.plot(rad_georef['azim [rad]'], var_mean, c='tab:green', linewidth=2,
            ls='', marker='s', markeredgecolor='g', alpha=0.4,
            label=labelm)
    # =========================================================================
    # Plot the radar variable offset
    # =========================================================================
    if var_offset != 0:
        ax.plot(rad_georef['azim [rad]'],
                np.full(rad_georef['azim [rad]'].shape, var_offset),
                c='k', linewidth=2.5, label=labelo)

    ax.set_thetagrids(np.arange(0, 360, 90))
    ax.xaxis.grid(ls='-')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_rlabel_position(-45)
    if var_name == 'PhiDP [deg]':
        ax.set_ylim([var_mean.mean()-dof, var_mean.mean()+dof])
        ax.set_yticks(np.arange(round(var_mean.mean()/dval)*dval-dof,
                                round(var_mean.mean()/dval)*dval+dof+1,
                                dval))
    else:
        ax.set_ylim([var_mean.mean()-dof, var_mean.mean()+dof])
        # ax.set_yticks(np.arange(round(var_mean.mean()/dval)*dval-dof,
        #                         round(var_mean.mean()/dval)*dval+dof+.1,
        #                         dval))
    angle = np.deg2rad(67.5)
    ax.legend(fontsize=15, loc="lower left",
              bbox_to_anchor=(.58 + np.cos(angle)/2, .4 + np.sin(angle)/2))
    ax.axes.set_aspect('equal')
    plt.tight_layout()


def plot_mfs(path_mfs, norm=True, vars_bounds=None, fig_size=None):
    """
    Plot the membership functions used in clutter classification.

    Parameters
    ----------
    path_mfs : str
        Location of the membership function files..
    norm : bool, optional
        Determines if the variables are normalised for a more comprehensive
        visualisation of the MFS. The default is True.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, LaTeX Varnames] between which radar variables are
        to be mapped.
    fig_size : list or tuple containing 2-element numbers, optional
        Width, height in inches. The default is None.
    """
    import os
    mfspk = {
        'ZHH': [[-10, 60], '$Z_H$ [dBZ]'],
        'sZhh': [[0, 20], r'$\sigma(Z_{H})$ [dBZ]'],
        'ZDR': [[-6, 6], '$Z_{DR}$ [dB]'],
        'sZdr': [[0, 5], r'$\sigma(Z_{DR}$) [dB]'],
        'Rhv': [[0, 1], r'$\rho_{HV}$ [-]'],
        'sRhv': [[0, .4], r'$\sigma(\rho_{HV})$ [-]'],
        'Pdp': [[0, 180], r'$\Phi_{DP}$ [deg]'],
        'sPdp': [[0, 180], r'$\sigma(\Phi_{DP})$ [deg]'],
        'Vel': [[-3, 3], 'V [m/s]'],
        'sVel': [[0, 5], r'$\sigma(V)$ [m/s]'],
        'LDR': [[-40, 10], 'LDR [dB]'],
        }
    if vars_bounds is not None:
        mfspk.update(vars_bounds)

    mfsp = {f[f.find('mf_')+3: f.find('_preci')]: np.loadtxt(f'{path_mfs}{f}')
            for f in sorted(os.listdir(path_mfs))
            if f.endswith('_precipi.dat')}
    mfsp = {k: v for k, v in sorted(mfsp.items()) if k in mfspk}
    mfsc = {f[f.find('mf_')+3: f.find('_clu')]: np.loadtxt(f'{path_mfs}{f}')
            for f in sorted(os.listdir(path_mfs))
            if f.endswith('_clutter.dat')}
    mfsc = {k: v for k, v in sorted(mfsc.items()) if k in mfspk}

    varsp = {k for k in mfsp.keys()}
    varsc = {k for k in mfsc.keys()}

    if len(varsp) % 2 == 0:
        ncols = int(len(varsp) / 2)
        nrows = len(varsp) // ncols
        if fig_size is None:
            fig_size = (18, 5)
    else:
        ncols = 3
        if len(varsp) % 3 == 0:
            nrows = (len(varsp) // ncols)
        else:
            nrows = (len(varsp) // ncols)+1
        if fig_size is None:
            fig_size = (18, 7.5)

    if varsp != varsc:
        raise TowerpyError('Oops!... The number of membership functions for'
                           + 'clutter and precipitation do not correspond.'
                           + 'Please check before continue.')

    if norm is True:
        mfs_prnorm = {k: np.array([val[:, 0], rut.normalisenan(val[:, 1])]).T
                      for k, val in mfsp.items()}
        mfs_clnorm = {k: np.array([val[:, 0], rut.normalisenan(val[:, 1])]).T
                      for k, val in mfsc.items()}

    f, ax = plt.subplots(nrows, ncols, sharey=True, figsize=fig_size)
    for a, (key, value) in zip(ax.flatten(), mfs_prnorm.items()):
        a.plot(value[:, 0], value[:, 1], c='tab:blue', label='PR')
        a.plot(mfs_clnorm[key][:, 0], mfs_clnorm[key][:, 1], label='CL',
               ls='dashed', c='tab:orange')
        # a.set_xlim(left=0)
        a.set_xlim(mfspk[key][0])
        a.tick_params(axis='both', labelsize=16)

        divider = make_axes_locatable(a)
        cax = divider.append_axes("top", size="15%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.set_facecolor('slategrey')

        at = AnchoredText(mfspk[key][1], loc='center',
                          prop=dict(size=18, color='white'), frameon=False)
        cax.add_artist(at)
        a.legend(fontsize=14)
    f.tight_layout()


def plot_zhah(rad_vars, r_ahzh, temp, coeff_a, coeff_b, coeffs_a, coeffs_b,
              temps, zh_lower_lim, zh_upper_lim, var2calc='ZH [dBZ]'):
    r"""
    Display the AH-ZH relation.

    Parameters
    ----------
    rad_vars : dict
        Dict containing radar variables to plot.
    r_ahzh : obj
        Results of the Attn_Refl_Relation class.
    temp: float
        Temperature, in :math:`^{\circ}C`, used to derive the coefficients
        according to [1]_. The default is 10.
    coeff_a, coeff_b: float
        Computed coefficients of the :math:`A_H(Z_H)` relationship.
    coeffs_a, coeffs_b: list or array
        Default coefficients of the :math:`A_H(Z_H)` relationship..
    temps : list or array
        Default values for the temperature.
    var2calc : str, optional
        Radar variable to be computed. The string has to be one of
        'AH [dB/km]' or 'ZH [dBZ]'. The default is 'ZH [dBZ]'.
    """
    tcksize = 14
    cmap = 'Spectral_r'
    n1 = mpc.LogNorm(vmin=1, vmax=1000)
    gridsize = 200
    ahzhii = np.arange(zh_lower_lim, zh_upper_lim, 0.05)
    ahzhlii = tpuc.xdb2x(ahzhii)
    ahzhi = coeff_a * ahzhlii ** coeff_b
    if var2calc == 'AH [dB/km]' and 'ZH [dBZ]' in rad_vars.keys():
        zh_all = rad_vars['ZH [dBZ]'].ravel()
        ah_all = rad_vars['AH [dB/km]'].ravel()
    elif 'AH [dB/km]' in r_ahzh.keys():
        zh_all = rad_vars['ZH [dBZ]'].ravel()
        ah_all = r_ahzh['AH [dB/km]'].ravel()
    if var2calc == 'ZH [dBZ]' and 'ZH [dBZ]' in rad_vars.keys():
        zh_all = rad_vars['ZH [dBZ]'].ravel()
        ah_all = rad_vars['AH [dB/km]'].ravel()
    # Plot the AH-ZH values
    fig, ax = plt.subplots()
    ax.plot(ahzhii, ahzhi, c='k', ls='--',
            label=f'$A_H={coeff_a:,.2e}Z_H^{{{coeff_b:,.2f}}}$')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("top", size="7%", pad="2%")
    hxb = ax.hexbin(np.ma.masked_invalid(zh_all),
                    np.ma.masked_invalid(ah_all), gridsize=gridsize,
                    mincnt=1, cmap=cmap, norm=n1)
    cb = fig.colorbar(hxb, cax=cax, extend='max',
                      orientation='horizontal')
    ax.set_xlim([-10, 60])
    ax.set_ylim([0, 2])
    cb.ax.tick_params(direction='in', labelsize=tcksize,)
    cb.ax.set_title('Counts', fontsize=tcksize)
    cax.xaxis.set_ticks_position("top")
    ax.set_xlabel('$Z_H$ [dBZ]')
    ax.set_ylabel('$A_H$ [dB/km]')
    ax.legend(loc='upper left')
    ax.grid()
    # Plot the linar interpolation of temp.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(rf'Linear Interpolation at T = {temp}$\degree$')
    ax = axs[0]
    ax.plot(coeffs_a, temps, '-ob')
    ax.plot(coeff_a, temp, 'ro')
    ax.set_xlabel('coeff a')
    ax.set_ylabel(r'Temp ${\degree}$C')
    ax = axs[1]
    ax.plot(coeffs_b, temps, '-ob')
    ax.plot(coeff_b, temp, 'ro')
    ax.set_xlabel('coeff b')
    fig.tight_layout()


def plot_ppidiff(rad_georef, rad_params, rad_var1, rad_var2, var2plot1=None,
                 var2plot2=None, diff_lims=[-10, 10, 1], mlyr=None, xlims=None,
                 ylims=None, vars_bounds=None, unorm=None, ucmap=None,
                 ucmap_diff=None, cb_ext=None, fig_title=None, fig_size=None):
    """
    Plot the difference between a radar variable from different dicts.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_var1 : dict
        Dict containing radar variables to plot.
    rad_var2 : dict
        Dict containing radar variables to plot.
    vars2plot : str, optional
        Keys of the radar variables to plot. Variables must have the same
        units. The default is None. This option will plot ZH or look for the
        'first' element in the rad_vars dict.
    diff_lims : 3-element tuple or list, optional
        Boundaries [min, max, step] used for mapping the difference plot.
        The default is [-10, 10, 1].
    mlyr : MeltingLayer Class, optional
        Plot the melting layer height. ml_top (float, int, list or np.array)
        and ml_bottom (float, int, list or np.array) must be explicitly
        defined. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    ucmap_diff : str of colormap, optional
        User-defined colormap used in the difference plot.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    fig_title : str, optional
        String to show in the plot title.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot1, rad_var1, vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    if var2plot1 is None:
        var2plot1 = v2p
    lpv2, bnd2, cmaph2, cmapext2, dnorm2, v2p2, normp2, cbtks_fmt2, tcks2 = pltparams(
        var2plot2, rad_var2, vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    if var2plot2 is None:
        var2plot2 = v2p2
    cmapp = cmaph.get(var2plot1[var2plot1.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
    if fig_title is None:
        if isinstance(rad_params['elev_ang [deg]'], str):
            dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
        else:
            dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} deg. -- "
        dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
        ptitle = dtdes1 + dtdes2
    else:
        ptitle = fig_title
    if mlyr is not None:
        if isinstance(mlyr.ml_top, (int, float)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_top, (np.ndarray, list, tuple)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        if isinstance(mlyr.ml_bottom, (int, float)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_bottom, (np.ndarray, list, tuple)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        mlt_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlt_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlb_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])
        mlb_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])
    # =========================================================================
    # Creates plots to visualise difference
    # =========================================================================
    mosaic = 'ABC'
    if fig_size is None:
        fig_size = (16, 5)
    fig_mos1 = plt.figure(figsize=fig_size, constrained_layout=True)
    ax_idx = fig_mos1.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_var1.items():
        if key == var2plot1:
            fzhna = ax_idx['A'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmapp,
                                           norm=normp)
            ax_idx['A'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['A'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['A'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['A'].legend(loc='upper left')
    if xlims is not None:
        ax_idx['A'].set_xlim(xlims)
    if ylims is not None:
        ax_idx['A'].set_ylim(ylims)
    # plt.colorbar(fzhna, ax=ax_idx['A']).ax.tick_params(labelsize=10)
    ax_idx['A'].grid(True)
    ax_idx['A'].axes.set_aspect('equal')
    ax_idx['A'].tick_params(axis='both', labelsize=10)
    for key, value in rad_var2.items():
        if key == var2plot2:
            fzhna = ax_idx['B'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmapp,
                                           norm=normp)
            ax_idx['B'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['B'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['B'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['B'].legend(loc='upper left')
    if xlims is not None:
        ax_idx['B'].set_xlim(xlims)
    if ylims is not None:
        ax_idx['B'].set_ylim(ylims)
    # plt.colorbar(fzhna, ax=ax_idx['B']).ax.tick_params(labelsize=10)
    if (var2plot1 == 'rhoHV [-]' or '[mm]' in var2plot1
       or '[mm/h]' in var2plot1):
        plt.colorbar(fzhna, ax=ax_idx['B'], ticks=tcks,
                     format=f'%.{cbtks_fmt}f')
    else:
        plt.colorbar(fzhna, ax=ax_idx['B'])
    ax_idx['B'].grid(True)
    ax_idx['B'].axes.set_aspect('equal')
    ax_idx['B'].tick_params(axis='both', labelsize=10)

    cmaph = 'tpylsc_div_dbu_rd'
    if ucmap_diff is not None:
        cmaph = ucmap_diff
    divnorm = mpl.colors.BoundaryNorm(
        rut.linspace_step(diff_lims[0], diff_lims[1], diff_lims[2]),
        mpl.colormaps[cmaph].N, extend='both')
    fzhna = ax_idx['C'].pcolormesh(rad_georef['grid_rectx'],
                                   rad_georef['grid_recty'],
                                   rad_var1[var2plot1]-rad_var2[var2plot2],
                                   shading='auto', cmap=cmaph, norm=divnorm)
    ax_idx['C'].set_title(f"{ptitle}" "\n"
                          + f"diff {var2plot1[var2plot1.find('['):]}")
    if mlyr is not None:
        ax_idx['C'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['C'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['C'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['C']).ax.tick_params(labelsize=10)
    ax_idx['C'].grid(True)
    ax_idx['C'].axes.set_aspect('equal')
    ax_idx['C'].tick_params(axis='both', labelsize=10)
