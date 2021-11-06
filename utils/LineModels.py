import numpy as np
from scipy.interpolate import interp2d, NearestNDInterpolator
from nbodykit.utils import DistributedArray
from nbodykit.lab import BigFileCatalog, MultipleSpeciesCatalog
from nbodykit.cosmology.cosmology import Cosmology
from pmesh.pm import ParticleMesh
from sfr import logSFR_Behroozi

gb_k_B = 1.38064852e-16 # Boltzmann constant in unit of erg K^-1
gb_L_sun = 3.83e33 # in unit of erg/s
gb_c = 2.99792458e5 # in units of km/s
gb_len_conv = 3.086e19 # conversion factor from Mpc to km

gb_h = 0.677

cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)

def H(z):
    return 100.* cosmo.h * cosmo.efunc(z)

# ###########################################        ###########################################   
# ########################################### MODELS ###########################################   
# ###########################################        ###########################################   
class ModelHI_A():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1/aa-1

        self.alp = (1+2*self.zz)/(2+2*self.zz)
        #self.mcut = 1e9*( 1.8 + 15*(3*self.aa)**8)
        self.mcut = 3e9*( 1 + 10*(3*self.aa)**8)
        ###self.normhalo = 3e5*(1+(3.5/self.zz)**6) 
        ###self.normhalo = 3e7 *(4+(3.5/self.zz)**6)
        self.normhalo = 8e5*(1+(3.5/self.zz)**6) 

        self.normsat = self.normhalo*(1.75 + 0.25*self.zz)
        self.normsat *= 0.5 #THis is to avoid negative masses
        
        #z=1
        if np.abs(self.zz-1.0)<0.1:
            self.alp = 0.76
            self.mcut = 2.6e10
            self.normhalo = 4.6e8
            self.normsat = self.normsat/5
        #z=0.5
        if np.abs(self.zz-0.5)<0.1:
            self.alp = 0.63
            self.mcut = 3.7e10
            self.normhalo = 9.8e8
            self.normsat = self.normsat/100
        #z=0
        if np.abs(self.zz - 0.0)<0.1:
            #print('Modify')
            self.alp = 0.49
            self.mcut = 5.2e10
            self.normhalo = 2.1e9
            self.normsat = self.normsat/100


    def assignline(self, halocat, cencat, satcat):
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(mHIhalo, mHIsat, satcat['GlobalID'].compute(), 
                                cencat.csize, cencat.comm)
        
        return mHIhalo, mHIcen, mHIsat
        
        
    def assignhalo(self, mhalo):
        xx  = mhalo/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normhalo
        return mHI

    def assignsat(self, msat):
        xx  = msat/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normsat
        return mHI
        

    def getinsat(self, mHIsat, satid, totalsize, localsize, comm):
       
        #print(comm.rank, np.all(np.diff(satid) >=0))
        #diff = np.diff(satid)
        #if comm.rank == 260: 
        #    print(satid[:-1][diff <0], satid[1:][diff < 0])
        da = DistributedArray(satid, comm)
        
        mHI = da.bincount(mHIsat, shared_edges=False)
        
        zerosize = totalsize - mHI.cshape[0]
        zeros = DistributedArray.cempty(cshape=(zerosize, ), dtype=mHI.local.dtype, comm=comm)
        zeros.local[...] = 0
        mHItotal = DistributedArray.concat(mHI, zeros, localsize=localsize)
        return mHItotal
        
    def assigncen(self, mHIhalo, mHIsat, satid, censize, comm):
        #Assumes every halo has a central...which it does...usually
        mHItotal = self.getinsat(mHIsat, satid, censize, mHIhalo.size, comm)
        return mHIhalo - mHItotal.local
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of HI
        '''

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
        rankweight       = sum([wt.sum() for wt in weights])
        totweight        = comm.allreduce(rankweight)
        for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3     
            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh
        
# ###########################################
class ModelHI_A2(ModelHI_A):
    '''Same as model A with a different RSD for satellites
    '''
    def __init__(self, aa):

        super().__init__(aa)
        
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity_HI']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos

# ###########################################        
class ModelHI_B():
    
    def __init__(self, aa, h=0.6776):

        self.aa = aa
        self.zz = 1/aa-1
        self.h = h
        #self.mcut = 1e9*( 1.8 + 15*(3*self.aa)**8 )
        self.mcut = 3e9*( 1 + 10*(3*self.aa)**8) 
        self.normhalo = 1
        #self.slope, self.intercept = np.polyfit([8.1, 11], [0.2, -1.], deg=1)


    def assignline(self, halocat, cencat, satcat):
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(cencat['Mass'].compute())
        
        mHIhalo = self.assignhalo(mHIcen, mHIsat, satcat['GlobalID'].compute(), 
                                halocat.csize, halocat.comm)
        return mHIhalo, mHIcen, mHIsat

    def assignhalo(self, mHIcen, mHIsat, satid, hsize, comm):
        #Assumes every halo has a central...which it does...usually
        mHItotal = self.getinsat(mHIsat, satid, hsize, mHIcen.size, comm)
        return mHIcen + mHItotal.local


    def getinsat(self, mHIsat, satid, totalsize, localsize, comm):
        da = DistributedArray(satid, comm)
        mHI = da.bincount(mHIsat, shared_edges=False)
        zerosize = totalsize - mHI.cshape[0]
        zeros = DistributedArray.cempty(cshape=(zerosize, ), dtype=mHI.local.dtype, comm=comm)
        zeros.local[...] = 0
        mHItotal = DistributedArray.concat(mHI, zeros, localsize=localsize)
        return mHItotal

    def _assign(self, mstellar):
        '''Takes in M_stellar and gives M_HI in M_solar
        '''
        mm = 3e8 #5e7
        f = 0.18 #0.35
        alpha = 0.4 #0.35
        mfrac = f*(mm/(mstellar + mm))**alpha
        mh1 = mstellar * mfrac
        return mh1
        
        

    def assignsat(self, msat, scatter=None):
        mstellar = self.moster(msat, scatter=scatter)/self.h
        mh1 = self._assign(mstellar)
        mh1 = mh1*self.h #* np.exp(-self.mcut/msat)
        return mh1


    def assigncen(self, mcen, scatter=None):
        mstellar = self.moster(mcen, scatter=scatter)/self.h
        mh1 = self._assign(mstellar)
        mh1 = mh1*self.h #* np.exp(-self.mcut/mcen)
        return mh1


    def moster(self, Mhalo, scatter=None):
        """ 
        moster(Minf,z): 
        Returns the stellar mass (M*/h) given Minf and z from Table 1 and                                                                  
        Eq. (2,11-14) of Moster++13 [1205.5807]. 
        This version now works in terms of Msun/h units,
        convert to Msun units in the function
        To get "true" stellar mass, add 0.15 dex of lognormal scatter.                    
        To get "observed" stellar mass, add between 0.1-0.45 dex extra scatter.         

        """
        z = self.zz
        Minf = Mhalo/self.h
        zzp1  = z/(1+z)
        M1    = 10.0**(11.590+1.195*zzp1)
        mM    = 0.0351 - 0.0247*zzp1
        beta  = 1.376  - 0.826*zzp1
        gamma = 0.608  + 0.329*zzp1
        Mstar = 2*mM/( (Minf/M1)**(-beta) + (Minf/M1)**gamma )
        Mstar*= Minf
        if scatter is not None: 
            Mstar = 10**(np.log10(Mstar) + np.random.normal(0, scatter, Mstar.size))
        return Mstar*self.h
        #                                                                                                                                          

    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos



    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of HI
        '''

        pm = ParticleMesh(BoxSize=bs,Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
        rankweight       = sum([wt.sum() for wt in weights])
        totweight        = comm.allreduce(rankweight)
        for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        if tofield: mesh = mesh.to_field()
        return mesh

# ###########################################
class ModelHI_C(ModelHI_A):
    '''Vanilla model with no centrals and satellites, only halo
    Halos have the COM velocity but do not have any dispersion over it
    '''
    def __init__(self, aa):

        super().__init__(aa)
        self.normsat = 0

        #self.alp = 1.0
        #self.mcut = 1e9
        #self.normhalo = 2e5*(1+3/self.zz**2) 
        #self.normhalo = 1.1e5*(1+4/self.zz) 
        self.alp = 0.9
        self.mcut = 1e10
        self.normhalo = 3.5e6*(1+1/self.zz) 


    def derivate(self, param, delta):
        if param == 'alpha':
            self.alp = (1+delta)*self.alp
        elif param == 'mcut':
            self.mcut = 10**( (1+delta)*np.log10(self.mcut))
        elif param == 'norm':
            self.mcut = 10**( (1+delta)*np.log10(self.normhalo))
        else:
            print('Parameter to vary not recongnized. Should be "alpha", "mcut" or "norm"')
            


    def assignline(self, halocat, cencat, satcat):
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(cencat['Mass'].compute())
        
        return mHIhalo, mHIcen, mHIsat
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of HI
        '''

        pm = ParticleMesh(BoxSize=bs,Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
        rankweight       = sum([wt.sum() for wt in weights])
        totweight        = comm.allreduce(rankweight)
        for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    
    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='halos', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        if tofield: mesh = mesh.to_field()
        return mesh

# ###########################################
class ModelHI_C2(ModelHI_C):
    '''Vanilla model with no centrals and satellites, only halo
    Halos have the COM velocity and a dispersion from VN18 added over it
    '''
    def __init__(self, aa):

        super().__init__(aa)
        self.vdisp = self._setupvdisp()


    def _setupvdisp(self):
        vzdisp0 = np.array([31, 34, 39, 44, 51, 54])
        vzdispal = np.array([0.35, 0.37, 0.38, 0.39, 0.39, 0.40])
        vdispz = np.arange(0, 6)
        vdisp0fit = np.polyfit(vdispz, vzdisp0, 1)
        vdispalfit = np.polyfit(vdispz, vzdispal, 1)
        vdisp0 = self.zz * vdisp0fit[0] + vdisp0fit[1]
        vdispal = self.zz * vdispalfit[0] + vdispalfit[1]
        return lambda M: vdisp0*(M/1e10)**vdispal
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        dispersion = np.random.normal(0, self.vdisp(halocat['Mass'].compute())).reshape(-1, 1)
        hvel = halocat['Velocity']*los + dispersion*los
        hrsdpos = halocat['Position']+ hvel*rsdfac
        
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos

    
# ###########################################
class ModelHI_D():
    '''Vanilla model with no centrals and satellites, only halo
    Halos have the COM velocity and a dispersion from VN18 added over it
    '''
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1/aa-1
        self.alp   = (1.+2*self.zz)/(2.+2*self.zz)
        self.mcut   = 6e10*np.exp(-0.75*self.zz) + 1
        self.normhalo  = 1.7e9/(1+self.zz)**(5./3.)
        self.nsatfhalo = 0.1
         
    def fsat_h1(self, mhalo):
        logmass = np.log10(mhalo)
        mminf = 9.5  #10 - 0.2*self.zz
        mhalf = 12.8 #13 - 0.1*self.zz
        fsat = 0.5/(mhalf-mminf)**2 * (logmass-mminf)**2
        fsat[logmass < mminf] = 0
        fsat[fsat > 0.8] = 0.8
        return fsat

    
    def nsat_h1(self, mhalo):
        return  ((1 + self.nsatfhalo*mhalo / self.mcut)**0.5).astype(int) #2 #int(mhalo*0 + 2)
            
    def vdisp(self, mhalo):
        h = cosmo.efunc(self.zz)
        return 1100. * (h * mhalo / 1e15) ** 0.33333
    
    
    def assignline(self, halocat, cencat, satcat):
        if halocat.comm.rank == 0:
            if cencat is not None: print("\nCencat not used")
            if satcat is not None: print("\nSatcat not used")
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = self.assignsat(halocat['Mass'].compute(), mHIhalo)
        mHIcen = self.assigncen(mHIhalo, mHIsat)
        #now repeat satellite catalog and reduce mass
        nsat = self.nsat_h1(halocat['Mass'].compute())
        mHIsat = np.repeat(mHIsat/nsat, nsat, axis=0)
        return mHIhalo, mHIcen, mHIsat
        
        
    def assignhalo(self, mhalo):        
        xx  = (mhalo + 1e-30)/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1.0/xx)
        mHI*= self.normhalo
        return mHI

    
    def assignsat(self, mhalo, mh1halo):
        frac  = self.fsat_h1(mhalo)
        mHI = mh1halo*frac
        return mHI
        
        
    def assigncen(self, mHIhalo, mHIsat):
        #Assumes every halo has a central...which it does...usually
        return mHIhalo - mHIsat
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = hrsdpos #cencat['Position']+cencat['Velocity']*los * rsdfac
        #now satellites
        nsat = self.nsat_h1(halocat['Mass'].compute())
        dispersion = np.random.normal(0, np.repeat(self.vdisp(halocat['Mass'].compute()), nsat)).reshape(-1, 1)*los
        hvel = np.repeat(halocat['Velocity'].compute(), nsat, axis=0)*los 
        hveldisp = hvel + dispersion
        srsdpos = np.repeat(halocat['Position'].compute(), nsat, axis=0) + hveldisp*rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of HI
        '''

        pm = ParticleMesh(BoxSize=bs,Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
        rankweight       = sum([wt.sum() for wt in weights])
        totweight        = comm.allreduce(rankweight)
        for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

# ###########################################
class ModelHI_D2():
    '''Vanilla model with no centrals and satellites, only halo
    Halos have the COM velocity and a dispersion from VN18 added over it
    '''
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1/aa-1
        self.alp   = (1.+2*self.zz)/(2.+2*self.zz)
        self.mcut   = 6e10*np.exp(-0.75*self.zz) + 1
        self.normhalo  = 1.7e9/(1+self.zz)**(5./3.)
        self.nsat = 10
         
    def fsat_h1(self, mhalo):
        logmass = np.log10(mhalo)
        mminf = 9.5  #10 - 0.2*self.zz
        mhalf = 12.8 #13 - 0.1*self.zz
        fsat = 0.5/(mhalf-mminf)**2 * (logmass-mminf)**2
        fsat[logmass < mminf] = 0
        fsat[fsat > 0.8] = 0.8
        return fsat

    
    def nsat_h1(self, mhalo):
        return 2 #int(mhalo*0 + 2)
            
    def vdisp(self, mhalo):
        h = cosmo.efunc(self.zz)
        return 1100. * (h * mhalo / 1e15) ** 0.33333
    
    
    def assignline(self, halocat, cencat, satcat):
        if halocat.comm.rank == 0:
            if cencat is not None: print("\nCencat not used")
            if satcat is not None: print("\nSatcat not used")
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = self.assignsat(halocat['Mass'].compute(), mHIhalo)
        mHIcen = self.assigncen(mHIhalo, mHIsat)
        return mHIhalo, mHIcen, mHIsat
        
        
    def assignhalo(self, mhalo):        
        xx  = (mhalo + 1e-30)/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1.0/xx)
        mHI*= self.normhalo
        return mHI

    
    def assignsat(self, mhalo, mh1halo):
        frac  = self.fsat_h1(mhalo)
        mHI = mh1halo*frac
        return mHI
        
        
    def assigncen(self, mHIhalo, mHIsat):
        #Assumes every halo has a central...which it does...usually
        return mHIhalo - mHIsat
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = hrsdpos 
        #now for satellites, return only the dispersion and factor it into account while painting
        srsdpos = self.vdisp(halocat['Mass'].compute()).reshape(-1, 1)*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of HI
        '''

        pm = ParticleMesh(BoxSize=bs,Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
        rankweight       = sum([wt.sum() for wt in weights])
        totweight        = comm.allreduce(rankweight)
        for wt in weights: wt /= totweight/float(nc)**3            

        lay = pm.decompose(positions[0])
        mesh.paint(positions[0], mass=weights[0], layout=lay, hold=True)
        if len(positions) > 1:
            for i in range(self.nsat):
                shift = np.random.normal(0, positions[1])
                pos = positions[0] + shift
                lay = pm.decompose(pos)
                mesh.paint(pos, mass=weights[1]/self.nsat, layout=lay, hold=True)
            
        return mesh

# ###########################################
class ModelCII_A():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.nu_line = 1902e9
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CII = 0.8475
        self.b_CII = 7.2203

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        lCII = np.power(10., self.a_CII * logSFR + self.b_CII) * np.power(10., 0.37*np.random.randn(len(logMhalo))) 
        # log10(L) = a*log10(SFR) + b + Normal(0, 0.37)
        lCII[logSFR == -1000.] = 0.
        return self.L_fac * lCII
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of the Line
        '''
        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)
            
#         mesh = mesh/mesh.cmean()

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of the Line
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)
        for cat in catalogs: cat[weight] /= totweight/float(nc)**3
            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs, Nmesh=[nc,nc,nc], position=position, weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh

# ###########################################
class ModelCII_B():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.nu_line = 1902e9
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CII = 1.0000
        self.b_CII = 6.9647

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        lCII = np.power(10., self.a_CII * logSFR + self.b_CII) * np.power(10., 0.37*np.random.randn(len(logMhalo))) 
        lCII[logSFR == -1000.] = 0.
        return self.L_fac * lCII
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of Line
        '''

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh
    
# ###########################################   
class ModelCII_C():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.nu_line = 1902e9
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CII = 0.8727
        self.b_CII = 6.7250

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        lCII = np.power(10., self.a_CII * logSFR + self.b_CII) * np.power(10., 0.37*np.random.randn(len(logMhalo))) 
        lCII[logSFR == -1000.] = 0.
        return self.L_fac * lCII
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of Line
        '''

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh
    
# ########################################### 
class ModelCII_D():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.nu_line = 1902e9
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CII = 0.9231
        self.b_CII = 6.5234

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        lCII = np.power(10., self.a_CII * logSFR + self.b_CII) * np.power(10., 0.37*np.random.randn(len(logMhalo))) 
        lCII[logSFR == -1000.] = 0.
        return self.L_fac * lCII
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of Line
        '''

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh

# ########################################### 
class ModelCO10():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.delta_MF = 1.0
        self.J = 1.0
        self.nu_line = self.J*115.27e9 # in unit of Hz for CO
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CO = 1.27
        self.b_CO = -1.0

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        L_IR = np.power(10., logSFR)/self.delta_MF * np.power(10., 10.)
        L_IR[logSFR == -1000.] = 0.
        Lprime_CO = np.power(10., (np.log10(L_IR)-self.b_CO)/self.a_CO) * np.power(10., 0.37*np.random.randn(len(logMhalo))) #in unit of K km/s pc^2
        #log10(LCO_prime) = (log10(L_IR) - b) / a + Normal(0, 0.37)
        L_CO = 4.9 * 1.0e-5 * self.J**3. * Lprime_CO #in unit of L_sun
        return self.L_fac * L_CO
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of the Line
        '''
        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)
            
#         mesh = mesh/mesh.cmean()
        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh

# ########################################### 
class ModelCO21():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.delta_MF = 1.0
        self.J = 2.0
        self.nu_line = self.J*115.27e9 # in unit of Hz for CO
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CO = 1.11
        self.b_CO = 0.6

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        L_IR = np.power(10., logSFR)/self.delta_MF * np.power(10., 10.)
        L_IR[logSFR == -1000.] = 0.
        Lprime_CO = np.power(10., (np.log10(L_IR)-self.b_CO)/self.a_CO) * np.power(10., 0.37*np.random.randn(len(logMhalo)))
        #log10(LCO_prime) = (log10(L_IR) - b)/a + Normal(0, 0.37)
        #in unit of K km/s pc^2
        L_CO = 4.9 * 1.0e-5 * self.J**3. * Lprime_CO #in unit of L_sun
        return self.L_fac * L_CO
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of Line
        '''

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh
    
# ########################################### 
class ModelCO32():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.delta_MF = 1.0
        self.J = 3.0
        self.nu_line = self.J*115.27e9 # in unit of Hz for CO
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CO = 1.18
        self.b_CO = 0.1

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        L_IR = np.power(10., logSFR)/self.delta_MF * np.power(10., 10.)
        L_IR[logSFR == -1000.] = 0.
        Lprime_CO = np.power(10., (np.log10(L_IR)-self.b_CO)/self.a_CO) * np.power(10., 0.37*np.random.randn(len(logMhalo)))  
        #in unit of K km/s pc^2
        L_CO = 4.9 * 1.0e-5 * self.J**3. * Lprime_CO #in unit of L_sun
        return self.L_fac * L_CO
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of Line
        '''

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3         

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh
    
# ########################################### 
class ModelCO43():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.delta_MF = 1.0
        self.J = 4.0
        self.nu_line = self.J*115.27e9 # in unit of Hz for CO
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CO = 1.09
        self.b_CO = 1.2

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        L_IR = np.power(10., logSFR)/self.delta_MF * np.power(10., 10.)
        L_IR[logSFR == -1000.] = 0.
        Lprime_CO = np.power(10., (np.log10(L_IR)-self.b_CO)/self.a_CO) * np.power(10., 0.37*np.random.randn(len(logMhalo))) 
        #in unit of K km/s pc^2
        L_CO = 4.9 * 1.0e-5 * self.J**3. * Lprime_CO #in unit of L_sun
        return self.L_fac * L_CO
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of Line
        '''

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh
    
# ########################################### 
class ModelCO54():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.delta_MF = 1.0
        self.J = 5.0
        self.nu_line = self.J*115.27e9 # in unit of Hz for CO
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CO = 1.05
        self.b_CO = 1.8

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        L_IR = np.power(10., logSFR)/self.delta_MF * np.power(10., 10.)
        L_IR[logSFR == -1000.] = 0.
        Lprime_CO = np.power(10., (np.log10(L_IR)-self.b_CO)/self.a_CO) * np.power(10., 0.37*np.random.randn(len(logMhalo))) 
        #in unit of K km/s pc^2
        L_CO = 4.9 * 1.0e-5 * self.J**3. * Lprime_CO #in unit of L_sun
        return self.L_fac * L_CO
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of Line
        '''

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh
    
# ########################################### 
class ModelCO65():
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1./aa - 1.

        self.mcut = 1.0e9
        
        self.delta_MF = 1.0
        self.J = 6.0
        self.nu_line = self.J*115.27e9 # in unit of Hz for CO
        self.L_fac = 1e6 / (8.*np.pi*gb_k_B*H(self.zz)) * (gb_c/self.nu_line)**3. * (1.+self.zz)**2. * gb_len_conv**-2. * gb_L_sun * gb_h**3.
        
        self.a_CO = 1.04
        self.b_CO = 2.2

    def assignline(self, halocat, cencat, satcat):
        haloL = self.assignhalo(halocat['Mass'].compute())
        satL = self.assignsat(satcat['Mass'].compute())
        cenL = self.assigncen(cencat['Mass'].compute())
        return haloL, satL, cenL
    
    def assignhalo(self, mhalo):
        logMhalo  = np.log10(mhalo[self.mcut < mhalo]/gb_h)
        logSFR = logSFR_Behroozi(z=self.zz, logMList=logMhalo)
        L_IR = np.power(10., logSFR)/self.delta_MF * np.power(10., 10.)
        L_IR[logSFR == -1000.] = 0.
        Lprime_CO = np.power(10., (np.log10(L_IR)-self.b_CO)/self.a_CO) * np.power(10., 0.37*np.random.randn(len(logMhalo))) 
        #in unit of K km/s pc^2
        L_CO = 4.9 * 1.0e-5 * self.J**3. * Lprime_CO #in unit of L_sun
        return self.L_fac * L_CO
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, positions, weights):
        '''use this to create mesh of Line
        '''

        pm = ParticleMesh(BoxSize=bs, Nmesh=[nc,nc,nc])
        mesh = pm.create(mode='real', value=0)
        comm = pm.comm
        
#         rankweight       = sum([wt.sum() for wt in weights])
#         totweight        = comm.allreduce(rankweight)
#         for wt in weights: wt /= totweight/float(nc)**3            

        for i in range(len(positions)):
            lay = pm.decompose(positions[i])
            mesh.paint(positions[i], mass=weights[i], layout=lay, hold=True)

        return mesh

    def createmesh_catalog(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass', tofield=False):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)
        if tofield: mesh = mesh.to_field()
        return mesh