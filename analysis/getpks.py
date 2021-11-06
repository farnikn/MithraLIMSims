import numpy as np
import re, os, sys, yaml
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, MultipleSpeciesCatalog, FFTPower
from nbodykit     import setup_logging
from mpi4py       import MPI
sys.path.append('../utils/')
import LineModels
# enable logging, we have some clue what's going on.
setup_logging('info')

#Get model as parameter
#Get parameter file
cfname = sys.argv[1]

with open(cfname, 'r') as ymlfile:
    args = yaml.load(ymlfile, Loader=yaml.FullLoader)

#
nc = args['nc']
bs = args['bs']
alist = args['alist']

#Global, fixed things
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm
if rank == 0: print(args, flush=True)


#Which model & configuration to use to assign lines
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

#

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


def calc_pk1d(aa, linemesh, outfolder):
    '''Compute the 1D redshift-space P(k) for the lines'''
    if rank==0: print('Calculating pk1d', flush=True)
    pkll = FFTPower(linemesh, mode='1d', dk=None, kmin=0.0).power
    # Extract the quantities we want and write the file.
    kk   = pkll['k']
    sn   = pkll.attrs['shotnoise']
    pk   = np.abs(pkll['power'])
    if rank==0:
        outputname = os.path.basename(os.path.normpath(outfolder))
        outputname = outputname.rsplit('_', 1)[0]
        fout = open(outfolder + outputname + "_pks_1d_{:6.4f}.txt".format(aa), "w")
        fout.write("# Subtracting SN={:15.5e}.\n".format(sn))
        fout.write("# {:>8s} {:>15s}\n".format("k","Pk_0_HI"))
        for i in range(kk.size):
            fout.write("{:10.5f} {:15.5e}\n".format(kk[i],pk[i]-sn))
        fout.close()


def calc_pkmu(aa, linemesh, outfolder, los=[0,0,1], Nmu=int(4)):
    '''Compute the redshift-space P(k) for the lines in mu bins'''
    if rank==0: print('Calculating pkmu', flush=True)
    pkll = FFTPower(linemesh, mode='2d', dk=None, kmin=0.0, Nmu=Nmu, los=los).power
    # Extract what we want.
    kk = pkll.coords['k']
    sn = pkll.attrs['shotnoise']
    pk = pkll['power']
    if rank==0: print('For mu-bins', pkll.coords['mu'], flush=True)
    # Write the results to a file.
    if rank==0:
        outputname = os.path.basename(os.path.normpath(outfolder))
        outputname = outputname.rsplit('_', 1)[0]
        fout = open(outfolder + outputname + "_pks_mu_{:02d}_{:06.4f}.txt".format(Nmu, aa), "w")
        fout.write("# Redshift space power spectrum in mu bins.\n")
        fout.write("# Subtracting SN={:15.5e}.\n".format(sn))
        ss = "# {:>8s}".format(r'k\mu')
        for i in range(pkll.shape[1]):
            ss += " {:15.5f}".format(pkll.coords['mu'][i])
        fout.write(ss+"\n")
        for i in range(1,pk.shape[0]):
            ss = "{:10.5f}".format(kk[i])
            for j in range(pk.shape[1]):
                ss += " {:15.5e}".format(np.abs(pk[i,j]-sn))
            fout.write(ss+"\n")
        fout.close()



def calc_pkll(aa, linemesh, outfolder, los=[0,0,1], Nmu=int(8)):
    '''Compute the redshift-space P_ell(k) for the line'''
    if rank==0: print('Calculating pkll', flush=True)
    pkll = FFTPower(linemesh, mode='2d', Nmu=Nmu, los=los, dk=None, kmin=0.0, poles=[0,2,4]).poles
    # Extract the quantities of interest.
    kk = pkll.coords['k']
    sn = pkll.attrs['shotnoise']
    P0 = pkll['power_0'].real - sn
    P2 = pkll['power_2'].real
    P4 = pkll['power_4'].real
    # Write the results to a file.
    if rank==0:
        outputname = os.path.basename(os.path.normpath(outfolder))
        outputname = outputname.rsplit('_', 1)[0]
        fout = open(outfolder + outputname + "_pks_ll_{:06.4f}.txt".format(aa), "w")
        fout.write("# Redshift space power spectrum multipoles.\n")
        fout.write("# Subtracting SN={:15.5e} from monopole.\n".format(sn))
        fout.write("# {:>8s} {:>15s} {:>15s} {:>15s}\n".\
                   format("k","P0","P2","P4"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:15.5e} {:15.5e} {:15.5e}\n".\
                       format(kk[i],P0[i],P2[i],P4[i]))
        fout.close()


def calc_bias(aa, linemesh, outfolder):
    '''Compute the bias(es) for the line'''
    if rank==0: print('Calculating bias', flush=True)
    if rank==0:
        print("Processing a={:.4f}...".format(aa), flush=True)
        print('Reading DM mesh...', flush=True)
    dm    = BigFileMesh(args['matterfile']%(aa),'N1024').paint()
    dm   /= dm.cmean()
    if rank==0: print('Computing DM P(k)...', flush=True)
    pkmm  = FFTPower(dm, mode='1d', dk=None, kmin=0.0).power
    k, pkmm= pkmm['k'], pkmm['power']  # Ignore shotnoise.
    if rank==0: print('Done.', flush=True)

    pkll = FFTPower(linemesh, mode='1d', dk=None, kmin=0.0).power
    kk = pkll.coords['k']

    pkll = pkll['power'] - pkll.attrs['shotnoise']
    pklm = FFTPower(linemesh, second=dm, mode='1d', dk=None, kmin=0.0).power['power']
    if rank==0: print('Done.', flush=True)
    # Compute the biases.
    b1x = np.abs(pklm/(pkmm+1e-10))
    b1a = np.abs(pkll/(pkmm+1e-10))**0.5
    if rank==0: print("Finishing processing a={:.4f}.".format(aa), flush=True)

    if rank==0:
        outputname = os.path.basename(os.path.normpath(outfolder))
        outputname = outputname.rsplit('_', 1)[0]
        fout = open(outfolder + outputname + "_bias_{:6.4f}.txt".format(aa), "w")
        fout.write("# {:>8s} {:>10s} {:>10s} {:>15s}\n".\
                   format("k","b1_x","b1_a","Pkmm"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:10.5f} {:10.5f} {:15.5e}\n".\
                       format(kk[i],b1x[i],b1a[i],pkmm[i].real))
        fout.close()
        
        
def calc_stochasticity(aa, linemesh, outfolder):
    '''Compute the stochasticity for the line'''
    if rank==0: print('Calculating stochasticity', flush=True)
    if rank==0:
        print("Processing a={:.4f}...".format(aa), flush=True)
        print('Reading DM mesh...', flush=True)
        
    dm    = BigFileMesh(args['matterfile']%(aa),'N1024').paint()
    dm   /= dm.cmean()
    if rank==0: print('Computing DM P(k)...', flush=True)
    pkmm  = FFTPower(dm, mode='1d', dk=None, kmin=0.0).power
    k, pkmm= pkmm['k'], pkmm['power']  # Ignore shotnoise.
    if rank==0: print('Done.', flush=True)

    pkll = FFTPower(linemesh, mode='1d', dk=None, kmin=0.0).power
    kk = pkll.coords['k']
    pkll = pkll['power'] - pkll.attrs['shotnoise']
    pklm = FFTPower(linemesh, second=dm, mode='1d', dk=None, kmin=0.0).power['power']
    if rank==0: print('Done.', flush=True)
        
    # Compute the stochasticity
    b1x = np.abs(pklm/(pkmm+1e-10))
    pkeps = pkll - 2.*b1x*pklm + b1x**2*pkmm
    
    if rank==0: print("Finishing processing a={:.4f}.".format(aa), flush=True)

    if rank==0:
        outputname = os.path.basename(os.path.normpath(outfolder))
        outputname = outputname.rsplit('_', 1)[0]
        fout = open(outfolder + outputname + "_stoch_{:6.4f}.txt".format(aa), "w")
        fout.write("# {:>8s} {:>10s} {:>15s} {:>15s} {:>15s} {:>15s}\n".\
                   format("k", "b1_x", "Pkll", "Pklm", "Pkmm", "Pkeps"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:10.5f} {:10.5f} {:15.5e} {:15.5e} {:15.5e}\n".\
                       format(kk[i], b1x[i], pkll[i].real, pklm[i].real, pkmm[i].real, pkeps[i].real))
        fout.close()


if __name__=="__main__":

    if rank==0: print('Starting Analysis', flush=True)

    for aa in alist:

        if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1), flush=True)

        for model in args['models']:
            linemodel = modeldict[model]
            modelname = model
            mode = modedict[model]
            
            #Path to save the output here
            outfolder = args['outfolder']%aa + 'pks/%s_N%04d/'%(modelname, nc)
            if rank == 0: print(outfolder, flush=True)
            for folder in [args['outfolder']%aa + '/pks/', outfolder]:
                try:  os.makedirs(folder)
                except : pass

            los = [0,0,1]
            try:
                linemesh = BigFileMesh(args['linemesh']%(aa, nc), modelname)
                linemeshz = BigFileMesh(args['linemeshz']%(aa, nc), modelname)
                
            except Exception as e:
                print('\nException occured : ', e, flush=True)

                mpart, Lbox, rsdfac, acheck = read_conversions(args['headerfilez']%aa)
                if np.abs(acheck-aa)>1e-4:
                    raise RuntimeError("Read a={:f}, expecting {:f}.".format(acheck,aa))
                if np.abs(Lbox-bs)>1e-4:
                    raise RuntimeError("Read L={:f}, expecting {:f}.".format(Lbox,bs))
                if rank == 0: print('Mass of the particle : %0.2e'%mpart, flush=True)
                
                halocat = BigFileCatalog(args['halofilez']%aa, dataset=args['halodataset'])
                halocat['Mass'] = halocat['Length'].compute() * mpart
                cencat = BigFileCatalog(args['cenfilez']%aa, dataset=args['cendataset'])
                satcat = BigFileCatalog(args['satfilez']%aa, dataset=args['satdataset'])
           
                linemodelc = linemodel(aa)
                
                halocat_line, cencat_line, satcat_line = linemodelc.assignline(halocat, cencat, satcat)
                halocat_rsdpos, cencat_rsdpos, satcat_rsdpos = linemodelc.assignrsd(rsdfac, halocat, cencat, satcat, los=los)

                if rank == 0: print('Creating line mesh in real space for bias', flush=True)
                positions = [halocat['Position']]
                weights = [halocat_line]
                linemesh = linemodelc.createmesh(bs, nc, positions, weights)

                if rank == 0: print('Creating line mesh in redshift space', flush=True)
                if mode=='halos':
                    positions = [halocat_rsdpos]
                    weights = [halocat_line]
                if mode=='galaxies':
                    positions = [cencat_rsdpos, satcat_rsdpos]
                    weights = [cencat_line, satcat_line]
                linemeshz = linemodelc.createmesh(bs, nc, positions, weights)

                
            calc_pk1d(aa, linemesh, outfolder)
            calc_bias(aa, linemesh, outfolder)
            calc_stochasticity(aa, linemesh, outfolder)

            calc_pkmu(aa, linemeshz, outfolder, los=los, Nmu=8)
            calc_pkll(aa, linemeshz, outfolder, los=los, Nmu=8)
                
    sys.exit(0)
