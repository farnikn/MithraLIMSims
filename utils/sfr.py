import numpy as np
from scipy.interpolate import interp2d, NearestNDInterpolator

def logSFR_Behroozi(z, logMList):
    # Halo masses should be in unit of M_sun.
    filename_SFR = '/mnt/home/fnikakhtar/lim/release-sfh_z0_z8_052913/sfr/sfr_release.dat'
    
    zp1, log10Mh, log10SFR, log10SM = np.loadtxt(filename_SFR, unpack=True)  
    numlines = np.size(log10SFR, 0) 
    num_z = len(np.unique(zp1))
    num_M = int(numlines/num_z)
    
    logMh_min, logMh_max = log10Mh.min(), log10Mh.max()
        
    z_arr = np.unique(zp1) - 1.
    logM_arr = np.unique(log10Mh)
    logSFR_arr = np.zeros((num_z, num_M))
    
    i = 0
    for q in range(num_M):
        for p in range(num_z):
            logSFR_arr[p, q] = log10SFR[i]
            i += 1
            
    logSFR_arr[logSFR_arr == -1000] = np.nan
    ZZ = np.ma.masked_invalid(logSFR_arr)
    
    XX_logM, YY_z = np.meshgrid(logM_arr, z_arr)
    
    X1_logM = XX_logM[~ZZ.mask]
    Y1_z = YY_z[~ZZ.mask]
    Z1_logSFR = ZZ[~ZZ.mask]
    
    interp_spline_SFR = NearestNDInterpolator(list(zip(X1_logM.flatten(), Y1_z.flatten())), Z1_logSFR.flatten())
    
    sfr_masked = interp_spline_SFR(logMList, z)
    
    z_max = zp1.max() - 1.
    
    if z <= z_max:
        result = sfr_masked
    else:
        result = sfr_masked + 0.2943*(z - 8)
        alt = 3.3847 - 0.2413*z
        result[alt < result] = alt

    result[result == 0] = -1000.
    return result