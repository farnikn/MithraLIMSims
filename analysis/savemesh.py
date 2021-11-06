import numpy as np
import re, os, sys, yaml
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, MultipleSpeciesCatalog, FFTPower, FieldMesh
from nbodykit     import setup_logging
from mpi4py       import MPI

sys.path.append('../utils/')
import LineModels
# enable logging, we have some clue what's going on.
setup_logging('info')

#Get parameter file
cfname = sys.argv[1]

with open(cfname, 'r') as ymlfile:
    args = yaml.load(ymlfile, Loader=yaml.FullLoader)


nc = args['nc']
bs = args['bs']
alist = args['alist']

#Global, fixed things
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm
scatter_1h = 4.270196
gb_h = 0.677
if rank == 0: print(args, flush=True)

modeldict = {'ModelHI_A':LineModels.ModelHI_A, 'ModelHI_B':LineModels.ModelHI_B, 
             'ModelHI_C':LineModels.ModelHI_C, 'ModelHI_D':LineModels.ModelHI_D, 'ModelHI_D2':LineModels.ModelHI_D2, 
             'ModelCII_A':LineModels.ModelCII_A, 'ModelCII_B':LineModels.ModelCII_B, 
             'ModelCII_C':LineModels.ModelCII_C, 'ModelCII_D':LineModels.ModelCII_D, 
             'ModelCO10':LineModels.ModelCO10, 'ModelCO21':LineModels.ModelCO21, 
             'ModelCO32':LineModels.ModelCO32, 'ModelCO43':LineModels.ModelCO43, 
             'ModelCO54':LineModels.ModelCO54, 'ModelCO65':LineModels.ModelCO65}

modedict = {'ModelHI_A':'galaxies', 'ModelHI_B':'galaxies', 'ModelHI_C':'halos', 'ModelHI_D':'galaxies', 'ModelHI_D2':'galaxies', 
            'ModelCII_A':'halos', 'ModelCII_B':'halos', 'ModelCII_C':'halos', 'ModelCII_D':'halos', 
            'ModelCO10':'halos', 'ModelCO21':'halos', 'ModelCO32':'halos', 'ModelCO43':'halos', 'ModelCO54':'halos', 'ModelCO65':'halos'} 


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
    
    
    
if __name__=="__main__":
    if rank==0: print('Starting', flush=True)

    for aa in alist:
        if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1), flush=True)
        mpart, Lbox, rsdfac, acheck = read_conversions(args['headerfilez']%aa)
        if np.abs(acheck-aa)>1e-4:
            raise RuntimeError("Read a={:f}, expecting {:f}.".format(acheck, aa))
        if np.abs(Lbox-bs)>1e-4:
            raise RuntimeError("Read L={:f}, expecting {:f}.".format(Lbox,bs))
        if rank == 0: print('Mass of the particle : %0.2e'%mpart, flush=True)

        halocat = BigFileCatalog(args['halofilez']%aa, dataset=args['halodataset'])
        halocat['Mass'] = halocat['Length'].compute() * mpart
        
        cencat = BigFileCatalog(args['cenfilez']%aa, dataset=args['cendataset'])
        satcat = BigFileCatalog(args['satfilez']%aa, dataset=args['satdataset'])

        for model in args['models']:
            linemodel = modeldict[model]
            modelname = model
            mode = modedict[model]

            linemodelc = linemodel(aa)
            
            los = [0,0,1]
            
            halocat_line, cencat_line, satcat_line = linemodelc.assignline(halocat, cencat, satcat)
            halocat_rsdpos, cencat_rsdpos, satcat_rsdpos = linemodelc.assignrsd(rsdfac, halocat, cencat, satcat, los=los)
            
            if rank == 0: print('Creating line mesh in real space for bias', flush=True)
                
            positions = [halocat['Position']]
            weights = [halocat_line]
            
            linemesh = linemodelc.createmesh(bs, nc, positions, weights)
            FieldMesh(linemesh).save(args['outfolder']%aa + '/linemesh3-N%04d/'%nc, dataset=modelname, mode='real')

            if rank == 0: print('Creating line mesh in redshift space', flush=True)
            
            if mode=='halos':
                positions = [halocat_rsdpos]
                weights = [halocat_line]
                
            if mode=='galaxies':
                positions = [cencat_rsdpos, satcat_rsdpos]
                weights = [cencat_line, satcat_line]
                
            linemeshz = linemodelc.createmesh(bs, nc, positions, weights)
            FieldMesh(linemeshz).save(args['outfolder']%aa + '/linemeshz3-N%04d/'%nc, dataset=modelname, mode='real')

            if rank == 0: print('Saved for %s'%modelname, flush=True)

    sys.exit(0)


