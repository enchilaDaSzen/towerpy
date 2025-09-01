"""Display an HTI plot from QVPs."""


import pickle
import towerpy as tp

RSITE = 'chenies'
elv = 'spel4'

if elv == 'spel8':
    ptype = 'VPs'
else:
    ptype = 'QVPs'
# WDIR = f'/../../../datasets/ukmo-nimrod/data/single-site/2020/chenies/spel4/'
WDIR = ('/home/enchiladaszen/Documents/sciebo/codes/github/towerpy/'
        + f'datasets/ukmo-nimrod/data/single-site/2020/{RSITE}/{elv}/')

with open(WDIR+f'{ptype.lower()}.tpy', 'rb') as f:
    rprofs = pickle.load(f)

with open(WDIR+f'mlyrs{ptype.lower()}.tpy', 'rb') as f:
    rmlyr = pickle.load(f)

# These objects were created with a previous version of Towerpy, so it is
# necessary to update the pof_type argument to agree with the latest release.
for i in rprofs:
    i.profs_type = ptype

tp.datavis.rad_display.plot_radprofiles(
    rprofs[4], rprofs[4].georef['profiles_height [km]'], colours=True,
    stats=None)
# %%


radb = tp.datavis.rad_interactive.hti_base(
    rprofs, mlyrs=rmlyr, stats='std', #ptype='fcontour',
    # var2plot='ZH [dBZ]',
    # var2plot='ZDR [dB]',
    var2plot='rhoHV [-]',
    # var2plot='PhiDP [deg]',
    # var2plot='V [m/s]',
    # var2plot='gradV [dV/dh]',
    # contourl='ZH [dBZ]',
    # ucmap='viridis',
    vars_bounds={'PhiDP [deg]': [35, 45, 11]},
    htiylim=[0, 8], tz='Europe/London')
radexpvis = tp.datavis.rad_interactive.HTI_Int()
radb.on_clicked(radexpvis.hzfunc)
