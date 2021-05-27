import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from spherical_bessel_transform import SphericalBesselTransform

kin_fid = np.logspace(-4.0,3.0,2000)
kout_fid = np.arange(0,0.5,0.01)

def make_window_matrix(windows, kin = kin_fid, kout = kout_fid):

    # Set up data vectors and transforms
    # The space is basically pvec = (p0,p2,p4) 
    # and the basis vectors are the unit vectors in that canonical representation

    nk  = len(kin)
    kells = np.concatenate( (kin,)*3 )
    
    sbt   = SphericalBesselTransform(kin,fourier=True)
    q0    = sbt.ydict[0]
    
    sbt_r = SphericalBesselTransform(q0,fourier=False)

    # Load window multipoles and turn into splines
    
    rwin, w0, w2, w4, w6, w8 = windows
    
    w0 = Spline(rwin,w0,ext=3)(q0)
    w2 = Spline(rwin,w2,ext=1)(q0)
    w4 = Spline(rwin,w4,ext=1)(q0)
    w6 = Spline(rwin,w6,ext=1)(q0)
    w8 = Spline(rwin,w8,ext=1)(q0)

    # Compute the response of each basis vector
    mat = []

    for ii, kell in enumerate(kells):
        
        pvec = np.zeros_like(kells); pvec[ii] = 1
        p0 = pvec[:nk]
        p2 = pvec[nk:2*nk]
        p4 = pvec[2*nk:]
    
        q0,xi0= sbt.sph(0,p0)
        q2,xi2= sbt.sph(2,p2); xi2 = Spline(q2,xi2)(q0)
        q4,xi4= sbt.sph(4,p4); xi4 = Spline(q4,xi4)(q0)
    
    
        xi0p   = xi0*w0 + 1./5.*xi2*w2  + 1./9.*xi4*w4
        xi2p   = xi0*w2 + xi2*(w0+2./7.*w2+2./7.*w4) + \
                xi4*(2./7.*w2+100./693.*w4 + 25./143.*w6)
        xi4p   = xi0*w4 + xi2*(18./35*w2 + 20./77*w4 + 45./143*w6)\
                 + xi4*(w0 + 20./77*w2 + 162./1001*w4 + 20./143*w6 + 490./2431*w8)
    
        k0,p0   = sbt_r.sph(0,4*np.pi*xi0p)
        k2,p2   = sbt_r.sph(2,4*np.pi*xi2p)
        k4,p4   = sbt_r.sph(4,4*np.pi*xi4p)
    
        thy0 = Spline(k0,p0,ext=1)
        thy2 = Spline(k2,p2,ext=1)
        thy4 = Spline(k4,p4,ext=1)
        dx   = kout[1]-kout[0]
    
        tmp0 = np.zeros_like(kout)
        tmp2 = np.zeros_like(kout)
        tmp4 = np.zeros_like(kout)

        for i in range(kout.size):
            ss     = np.linspace(kout[i]-dx/2,kout[i]+dx/2,100)
            ivol   = 3.0/((kout[i]+dx/2)**3-(kout[i]-dx/2)**3)
        
            tmp0[i]= np.trapz(ss**2*thy0(ss),x=ss)*ivol
            tmp2[i]= np.trapz(ss**2*thy2(ss),x=ss)*ivol
            tmp4[i]= np.trapz(ss**2*thy4(ss),x=ss)*ivol
    
        mat += [ list(tmp0) + list(tmp2) + list(tmp4), ]
    
    mat = np.array(mat)
    
    return mat