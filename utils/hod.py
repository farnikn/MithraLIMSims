import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp1d, interp2d
from scipy.special import erf
from time import time
from nbodykit.cosmology.cosmology import Cosmology

cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)


##Setup nfw sampling here
def getrvir(m, z):
    '''https://halotools.readthedocs.io/en/latest/_modules/halotools/empirical_models/phase_space_models/analytic_models/halo_boundary_functions.html
    '''
    rho_crit = 3 * 100**2 /(8 * math.pi * 43.007)
    rho_crit *= 1e10
    x = cosmo.Om(z) - 1.0
    delta = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2
    rho = rho_crit * delta
    radius = (m * 3.0 / 4.0 / np.pi / rho)**(1.0 / 3.0)
    return radius
    
def nfw(r, m, z, c=7):
    rvir = getrvir(m, z)
    Rs = rvir/c
    mfac = 4*np.pi*Rs**3 *(np.log(1+c) - c/(1+c))
    rho0 = m/mfac
    return rho0/((r+1e-10)/Rs * (1 + r/Rs)**2)


def get_nfw_r(c=7):
    '''Martin's code based on rejection sampling
    '''
    x = np.random.uniform() * c
    while (np.random.uniform() > 4*x/(1+x)/(1+x)):
        x = np.random.uniform() * c
    else:
        return(x/c)

def gcum(x):
    return np.log(1+x) - x/(1+x)
    
def cumnfw(r, c=7):
    '''cumulative pdf for nfw profile at scaled radius(by r_vir)=r
    Taken from https://halotools.readthedocs.io/en/latest/source_notes/empirical_models/phase_space_models/nfw_profile_source_notes.html
    '''
    return gcum(r*c)/gcum(c)

def ilogcdfnfw(c=7):
    '''Inverse cdf of nfw in log-scale
    '''
    rr = np.logspace(-4, 0, 1000)
    cdf = cumnfw(rr, c=c)
    lrr, lcdf = np.log(rr), np.log(cdf)
    return ius(lcdf, lrr)
    

def sampleilogcdf(n, ilogcdf, seed=100):
    '''Inverse cdf sampling in log-scale
    '''
    np.random.seed(seed)
    u = np.random.uniform(size=n)
    lu = np.log(u)
    lx = ilogcdf(lu)
    return np.exp(lx)


##HOD functions here
def ncen(mh, mcutc, sigma):
    '''Zheng HOD setup with base log_10, can change sigma when using values from papers
    '''
    return 0.5* (1 + erf ((np.log10(mh)-np.log10(mcutc))/sigma ))



def nsat_zheng(mh, m0, m1, alpha):
    return ((mh - m0)/m1) ** alpha

def nsat_martin(msat, mh, m1f, alpha):
    '''return number of satellites with mass greater than msat
    for host of mass mh and with normalization of 1 satellite at m1=m1f*mh
    '''
    return ( msat/(m1f*mh) ) ** alpha


def sample_powerlaw(n, xmin, xmax, alpha, seed=100):
    '''get 'n' samples from power law with cdf index 'alpha'
    between xmin, xmax
    '''
    np.random.seed(seed)
    y = np.random.uniform(size=n)
    return  (xmin**alpha + (xmax**alpha - xmin**alpha)*y)**(1/alpha)


def get_msat(mh, mmin, mmax, alpha, seed=100):
    '''alpha is the index of the cdf of (M_h/m_sat) where msat is what we want to sample
    '''
    n = mh.size
    smass = mh *sample_powerlaw(n, mmin/mh, mmax/mh, alpha, seed=seed)
    return smass



def mksat(nsat, 
    pos, vel, rvir, vdisp, conc, 
    vsat=0.5, seed=231
    ):
    """
    Satellite galaxy HOD model
    nsat : number of satellites for every halo
    pos : position of halos
    vel : velocity of halos
    vdisp : is the DM velocity dispersion of the halo.
    conc : is the DM halo concentration parameter 
    -- currently only for a single concentration
    rvir : is the radius of the DM halo
    vsat : is the fraction of velocity relative to the dispersion.
    
    Returns (spos, svel, hid), position, velocity and haloid of satellites.
    """

    np.random.seed(seed)
    totsat = nsat.sum()
    #hid = np.repeat(hindex, nsat).astype(int)
    hid = np.repeat(range(len(nsat)), nsat).astype(int)
    satpos = np.zeros((totsat, 3))
    satvel = np.zeros((totsat, 3))

    #samplings
    ilcdf =  ilogcdfnfw(conc)
    rr = sampleilogcdf(totsat, ilcdf, seed=seed)
    phi   = 2*np.pi*np.random.uniform(size=totsat)
    ctheta = -1 + 2*np.random.uniform(size=totsat)
    vsample = np.random.normal(size=totsat*3).reshape(totsat, 3)

    dr0 = rr*np.sqrt(1-ctheta*ctheta)*np.cos(phi)
    dr1 = rr*np.sqrt(1-ctheta*ctheta)*np.sin(phi)
    dr2 = rr*ctheta;
    
    spos = pos[hid] + np.stack((dr0, dr1, dr2), axis=-1) * rvir[hid].reshape(-1, 1)
    svel = vel[hid] + vsample * vdisp[hid].reshape(-1, 1) * vsat

    return spos, svel, hid




#
if __name__=="__main__":
    
    aa = 0.2000
    nhalo = 100
    hmass = np.logspace(10, 14, nhalo)
    hpos = np.random.uniform(size=nhalo*3).reshape(nhalo, 3)
    hvel = np.random.uniform(size=nhalo*3).reshape(nhalo, 3)
    rvir = np.ones(nhalo)
    vdisp = np.ones(nhalo)
    conc = 7
    
    mmin = 1e9*( 1.8 + 15*(3*aa)**8 ) * 0.1 # 0.1 is the lim from appendix
    nsat = ((mmin/hmass)**-0.75 * np.exp(-np.log10(hmass)/13)).astype('int')
    print(nsat)

    ##Time testing to compare with simplehod
    import simplehod
    begin = time()
    spos, svel = simplehod.mksat(3454, nsat=nsat, pos=hpos, vel=hvel, vdisp=vdisp, conc=7, rvir=rvir)
    print('simplehod time : ', time() - begin)

    begin = time()
    spos, svel, shid = mksat(nsat=nsat, pos=hpos, vel=hvel, vdisp=vdisp, conc=7, rvir=rvir)
    print('my time : ', time() - begin)
    print(shid)
