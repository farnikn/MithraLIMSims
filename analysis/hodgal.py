import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, ArrayCatalog
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.transform import HaloRadius, HaloVelocityDispersion
from nbodykit.cosmology.cosmology import Cosmology
from time import time
import os, sys
import yaml, re

#
sys.path.append('../utils/')
import hod

#Get parameter file
cfname = sys.argv[1]

with open(cfname, 'r') as ymlfile:
    args = yaml.load(ymlfile, Loader=yaml.FullLoader)


nc = args['nc']
bs = args['bs']
alist = args['alist']
#
#
#Global, fixed things
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm
if rank == 0: print(args, flush=True)


def read_conversions(db):
    """Read the conversion factors we need and check we have the right time."""
    mpart,Lbox,rsdfac,acheck = None,None,None,None
    with open(db+"/attr-v2","r") as ff:
        for line in ff.readlines():
            mm = re.search("MassTable.*\#HUMANE\s+\[\s*0\s+(\d*\.\d*)\s*0+\s+0\s+0\s+0\s+\]",line)
            if mm != None:
                mpart = float(mm.group(1)) * 1e10
            mm = re.search("BoxSize.*\#HUMANE\s+\[\s*(\d+)\s*\]",line)
            if mm != None:
                Lbox = float(mm.group(1))
            mm = re.search("RSDFactor.*\#HUMANE\s+\[\s*(\d*\.\d*)\s*\]",line)
            if mm != None:
                rsdfac = float(mm.group(1))
            mm = re.search("ScalingFactor.*\#HUMANE\s+\[\s*(\d*\.\d*)\s*\]",line)
            if mm != None:
                acheck = float(mm.group(1))
    if (mpart is None)|(Lbox is None)|(rsdfac is None)|(acheck is None):
        print(mpart,Lbox,rsdfac,acheck, flush=True)
        raise RuntimeError("Unable to get conversions from attr-v2.")
    return mpart, Lbox, rsdfac, acheck
    #


    
def make_galcat(aa, mmin, m1f, alpha=-1, censuff=None, satsuff=None, ofolder=None, seed=3333):
    '''Assign 0s to
    '''
    zz = 1/aa-1

    halocat = BigFileCatalog(args['halofilez']%aa, dataset=args['halodataset'])
    rank = halocat.comm.rank

    mpart, Lbox, rsdfac, acheck = read_conversions(args['headerfilez']%aa)
    halocat.attrs['BoxSize'] = [bs, bs, bs] 
    halocat.attrs['NC'] = nc

    ghid = halocat.Index.compute()
    halocat['GlobalIndex'] = ghid
    halocat['Mass'] = halocat['Length'] * mpart
    halocat['Position'] = halocat['Position']%bs # Wrapping positions assuming periodic boundary conditions
    rank = halocat.comm.rank
        
    halocat = halocat.to_subvolumes()

    if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%zz, flush=True)

    hmass = halocat['Mass'].compute()
    hpos = halocat['Position'].compute()
    hvel = halocat['Velocity'].compute()
    rvir = HaloRadius(hmass, cosmo, 1/aa-1).compute()/aa
    vdisp = HaloVelocityDispersion(hmass, cosmo, 1/aa-1).compute()
    ghid = halocat['GlobalIndex'].compute()

    print('In rank = %d, Catalog size = '%rank, hmass.size, flush=True)
    #Do hod    
    start = time()
    ncen = np.ones_like(hmass)
    nsat = hod.nsat_martin(msat = mmin, mh=hmass, m1f=m1f, alpha=alpha).astype(int)
    
    #Centrals
    cpos, cvel, gchid, chid = hpos, hvel, ghid, np.arange(ncen.size)
    spos, svel, shid = hod.mksat(nsat, pos=hpos, vel=hvel, 
                                 vdisp=vdisp, conc=7, rvir=rvir, vsat=0.5, seed=seed)
    gshid = ghid[shid]
    svelh1 = svel*2/3 + cvel[shid]/3.

    smmax = hmass[shid]/10.
    smmin = np.ones_like(smmax)*mmin
    mask = smmin > smmax/3. #If Mmin and Mmax are too close for satellites, adjust Mmin
    smmin[mask] = smmax[mask]/3.
    smass = hod.get_msat(hmass[shid], smmax, smmin, alpha)

    
    sathmass = np.zeros_like(hmass)
    tot = np.bincount(shid, smass)
    sathmass[np.unique(shid)] = tot

    cmass = hmass - sathmass    # assign remaining mass in centrals

    print('In rank = %d, Time taken = '%rank, time()-start, flush=True)
    print('In rank = %d, Number of centrals & satellites = '%rank, ncen.sum(), nsat.sum(), flush=True)
    print('In rank = %d, Satellite occupancy: Max and mean = '%rank, nsat.max(), nsat.mean(), flush=True)
    #
    #Save
    cencat = ArrayCatalog({'Position':cpos, 'Velocity':cvel, 'Mass':cmass,  'GlobalID':gchid, 
                           'Nsat':nsat, 'HaloMass':hmass}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])
    minid, maxid = cencat['GlobalID'].compute().min(), cencat['GlobalID'].compute().max()
    if minid < 0 or maxid < 0:
        print('before ', rank, minid, maxid, flush=True)
    cencat = cencat.sort('GlobalID')
    minid, maxid = cencat['GlobalID'].compute().min(), cencat['GlobalID'].compute().max() 
    if minid < 0 or maxid < 0:
        print('after ', rank, minid, maxid, flush=True)

    if censuff is not None:
        colsave = [cols for cols in cencat.columns]
        cencat.save(ofolder+'cencat'+censuff, colsave)
    

    satcat = ArrayCatalog({'Position':spos, 'Velocity':svel, 'Velocity_HI':svelh1, 'Mass':smass,  
                           'GlobalID':gshid, 'HaloMass':hmass[shid]}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])
    minid, maxid = satcat['GlobalID'].compute().min(), satcat['GlobalID'].compute().max() 
    if minid < 0 or maxid < 0:
        print('before ', rank, minid, maxid, flush=True)
    satcat = satcat.sort('GlobalID')
    minid, maxid = satcat['GlobalID'].compute().min(), satcat['GlobalID'].compute().max() 
    if minid < 0 or maxid < 0:
        print('after ', rank, minid, maxid, flush=True)

    if satsuff is not None:
        colsave = [cols for cols in satcat.columns]
        satcat.save(ofolder+'satcat'+satsuff, colsave)


if __name__=="__main__":
    
    for aa in alist[:]:
        #Parameters for populating with satellites
        #sat hod : N = (M_h/m1)**alpha
        mmin = 1e9*( 1.8 + 15*(3*aa)**8 ) * 0.1 #mcut * 0.1, 0.1 being mmin
        alpha = -0.8

        for m1fac in [0.03]:
            censuff ='' #suffix for central catalog
            satsuff ='' #suffix for satellite catalog

            make_galcat(aa=aa, mmin=mmin, m1f=m1fac, alpha=alpha, censuff=censuff, satsuff=satsuff, ofolder=args['outfolder']%aa)

    sys.exit(0)
