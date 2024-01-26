import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.special import sph_harm
import healpy as hp
import time

#################
#SOME CONSTANTS
#################

c = 299792458.
day_s = 86400.
Om_Earth = 2*np.pi/day_s

##############################################
#DETECTORS INFORMATION --> TRANSFORM IN DICTIONARIES JUST ANOTHER .PY FILE
#The general coordinate system is WGS-84 Earth Model, described in the documentation of lal_detector (https://lscsoft.docs.ligo.org/lalsuite/lal/group___l_a_l_detectors__h.html),
#and the document T980044 (https://dcc.ligo.org/public/0002/T980044/001/T980044-10.pdf).
##############################################


#LIGO-Hanford, LIGO-Livingston, Virgo
#https://dcc.ligo.org/public/0072/P000006/000/P000006-D.pdf
#https://journals.aps.org/prd/pdf/10.1103/PhysRevD.63.042003

u_H = np.array([-0.223891216, 0.799830697, 0.556905359])
v_H = np.array([-0.913978490, 0.0260953206, -0.404922650])
L_H = 4000
x_H = np.array([-2161414.92635999, -3834695.17889000, 4600350.22664000])

u_L = np.array([-0.954574615, -0.141579994, -0.262187738])
v_L = np.array([0.297740169, -0.487910627, -0.820544948])
L_L = 4000
x_L = np.array([-74276.04472380, -5496283.71970999, 3224257.01744000])

u_V = np.array([-0.7005, 0.2085, 0.6826])
v_V = np.array([-0.0538, -0.9691, 0.2408])
L_V = 3000
x_V = np.array([4546374., 842990., 4378577.])

#KAGRA
u_K = np.array([-0.3759040, -0.8361583, 0.3994189])
v_K = np.array([0.7164378, 0.01114076, 0.6975620])
L_K_x = 2*1513.2535
L_K_y = 2*1511.611
L_K = (L_K_x + L_K_y)/2
x_K = np.array([-3777336.024, 3484898.411, 3765313.697])

#LIGO-India
#https://dcc.ligo.org/DocDB/0167/T2000158/001/LIO_coordinateSystem.pdf->currently commented
u_I = np.array([0.38496278183, -0.39387275094, 0.83466634811])
v_I = np.array([0.89838844906, -0.04722636126, -0.43665531647])
L_I = 4000
x_I = np.array([1.34897115479e6, 5.85742826577e6, 2.12756925209e6])

#is the one in lal the most updated one?-> nope this is a place-holder
#u_I = np.array([-9.72097635269165039e-01, 2.34576612710952759e-01, -4.23695567519644101e-08])
#v_I = np.array([-5.76756671071052551e-02, -2.39010959863662720e-01, 9.69302475452423096e-01])
#L_I = 4000
#x_I = np.array([1450526.82294155, 6011058.39047265, 1558018.27884102])

#GEO600 #see below
x_G = np.array([3.856310e6, 0.666599e6, 5.019641e6])
u_G = np.array([-0.6261, -0.5522, 0.5506])
v_G = np.array([-0.4453, 0.8665, 0.2255])
L_G = 600



####################
#UNIT VECTORS
#Romano-Cornish notation, spherical coordinates.
####################

def n_hat(theta, phi):
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    n_x = st*cp
    n_y = st*sp
    n_z = ct

    return np.squeeze([n_x, n_y, n_z])#.T

#dn/d_th
def l_hat(theta, phi):
    
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    l_x = ct*cp
    l_y = ct*sp
    l_z = -st
    
    return np.squeeze([l_x, l_y, l_z])#.T

#d(n/th)/dphiz
def m_hat(theta, phi):
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    m_x = -sp
    m_y = cp
    m_z = cp*int(0)

    return np.squeeze([m_x, m_y, m_z])#.T

#all units vectors together
def unit_vectors(theta, phi):
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    n_x = st*cp
    n_y = st*sp
    n_z = ct
    
    l_x = ct*cp
    l_y = ct*sp
    l_z = -st
    
    m_x = -sp
    m_y = cp
    m_z = cp*int(0)

    n = np.squeeze([n_x, n_y, n_z])#.T
    l = np.squeeze([l_x, l_y, l_z])#.T
    m = np.squeeze([m_x, m_y, m_z])#.T
    
    return n, l, m

#polarization tensors
def e_pol (theta, phi):
    """
    Input: theta=[0,pi), phi=[0,2pi) must be lists 
    Output: e_cross, e_plus
    """
    
      
    _, ll, mm = unit_vectors(theta, phi)
    
    e_plus = np.array([np.array([ll[0]*ll[0], ll[0]*ll[1], ll[0]*ll[2]]), np.array([ll[1]*ll[0], ll[1]*ll[1], ll[1]*ll[2]]), np.array([ll[2]*ll[0],ll[2]*ll[1], ll[2]*ll[2]])])\
    -np.array([np.array([mm[0]*mm[0], mm[0]*mm[1], mm[0]*mm[2]]), np.array([mm[1]*mm[0], mm[1]*mm[1], mm[1]*mm[2]]), np.array([mm[2]*mm[0],mm[2]*mm[1], mm[2]*mm[2]])])
    e_cross = np.array([np.array([ll[0]*mm[0], ll[0]*mm[1], ll[0]*mm[2]]), np.array([ll[1]*mm[0], ll[1]*mm[1], ll[1]*mm[2]]), np.array([ll[2]*mm[0],ll[2]*mm[1], ll[2]*mm[2]])])\
    +np.array([np.array([mm[0]*ll[0], mm[0]*ll[1], mm[0]*ll[2]]), np.array([mm[1]*ll[0], mm[1]*ll[1], mm[1]*ll[2]]), np.array([mm[2]*ll[0],mm[2]*ll[1], mm[2]*ll[2]])])
    
    
    #e_cross = np.array(np.tensordot(ll,mm, axes = 0) + np.tensordot(mm,ll, axes = 0))
    #e_plus = np.array(np.tensordot(ll,ll, axes = 0) - np.tensordot(mm,mm, axes = 0))
    
    #e_cross = np.array(np.tensordot(ll,mm, axes = 0) + np.tensordot(mm,ll, axes = 0))
    #e_cross = np.array(list(map(np.tensordot,ll,mm,[0]*len(ll)))) + np.array(list(map(np.tensordot,ll,mm,[0]*len(ll))))



    return e_cross, e_plus

def e_pol_hats(ll, mm):
    """
    Input: l_hat ,m_hat
    Output: e_cross, e_plus
    """
    
    #e_plus = np.array([np.array([ll[0]*ll[0], ll[0]*ll[1], ll[0]*ll[2]]), np.array([ll[1]*ll[0], ll[1]*ll[1], ll[1]*ll[2]]), np.array([ll[2]*ll[0],ll[2]*ll[1], ll[2]*ll[2]])])\
    #-np.array([np.array([mm[0]*mm[0], mm[0]*mm[1], mm[0]*mm[2]]), np.array([mm[1]*mm[0], mm[1]*mm[1], mm[1]*mm[2]]), np.array([mm[2]*mm[0],mm[2]*mm[1], mm[2]*mm[2]])])
    #e_cross = np.array([np.array([ll[0]*mm[0], ll[0]*mm[1], ll[0]*mm[2]]), np.array([ll[1]*mm[0], ll[1]*mm[1], ll[1]*mm[2]]), np.array([ll[2]*mm[0],ll[2]*mm[1], ll[2]*mm[2]])])\
    #+np.array([np.array([mm[0]*ll[0], mm[0]*ll[1], mm[0]*ll[2]]), np.array([mm[1]*ll[0], mm[1]*ll[1], mm[1]*ll[2]]), np.array([mm[2]*ll[0],mm[2]*ll[1], mm[2]*ll[2]])])
    
    #e_plus = np.array(np.tensordot(ll,ll, axes = 0) - np.tensordot(mm,mm, axes = 0))
    #e_cross = np.array(np.tensordot(ll,mm, axes = 0) + np.tensordot(mm,ll, axes = 0))
    
    e_plus = np.einsum('i...,j...', ll, ll) - np.einsum('i...,j...', mm, mm)
    e_cross = np.einsum('i...,j...', ll, mm) + np.einsum('i...,j...', mm, ll)

    return e_plus, e_cross

def e_pol_hats_non_GR(nn, ll, mm):
    
    e_plus, e_cross = e_pol_hats(ll, mm)
    
    e_vec_x=np.einsum('i...,j...', ll, nn) + np.einsum('i...,j...', nn, ll)
    e_vec_y=np.einsum('i...,j...', mm, nn) + np.einsum('i...,j...', nn, mm)
    
    e_scal_breath = np.einsum('i...,j...', ll, ll) + np.einsum('i...,j...', mm, mm)
    e_scal_lon = np.sqrt(2)*np.einsum('i...,j...', nn, nn)
    
    return e_plus, e_cross, e_vec_x, e_vec_y, e_scal_breath, e_scal_lon


######################
#ANTENNA PATTERNS     --> Remove old functions!!!!
######################

#time transfer function for two-way tracking
#np.sinc(x) = np.sin(np.pi*x)/(np.pi*c)
def T_u (f, n_hat, u, L, mode = "small antenna limit"):
    """
    Evaluates the time transfer function
    
    Input:
    f -> frequency
    n_hat -> direction in the sky
    u -> arm unit vector with respect to Earth center (?)
    L -> arm length (meters) 
    mode -> "small antenna limits" by default, otherwise full expression
    
    Output:
    2-way track Time transfer function
    """

    if mode == "small antenna limit":
        T_u_small = 2*L/c
        return (T_u_small)
    
    else:
        T_u_full = (L/c)*np.exp(-2j*np.pi*f*L/c) \
        *(np.exp(-1j*np.pi*f*(L/c)*(1-np.dot(n_hat, u)))*np.sinc(f*(L/c)*(1+np.dot(n_hat,u))) \
          + np.exp(1j*np.pi*f*(L/c)*(1+np.dot(n_hat, u)))*np.sinc(f*(L/c)*(1-np.dot(n_hat,u))))

        return(T_u_full)

def T_u_small (f, n_hat, u, L):
    """
    Evaluates the time transfer function
    
    Input:
    f -> frequency
    n_hat -> direction in the sky
    u -> arm unit vector with respect to Earth center (?)
    L -> arm length (meters) 
    
    Output:
    2-way track Time transfer function
    """
    return 2*L/c    

def T_u_full (f, n_hat, u, L):
    """
    Evaluates the time transfer function
    
    Input:
    f -> frequency
    n_hat -> direction in the sky
    u -> arm unit vector with respect to Earth center (?)
    L -> arm length (meters) 
    
    Output:
    2-way track Time transfer function
    """
    n_dot_u = np.einsum('i,i...', u, n_hat)#np.tensordot(u, n_hat, axes = 1)
    phase = 1j*np.pi*f*L/c
    f_phase = f*(L/c)
    
    T_u_full = (L/c)*np.exp(-2*phase) \
    *(np.exp(-phase*(1 - n_dot_u))*np.sinc(f_phase*(1 + n_dot_u)) \
    + np.exp(phase*(1 + n_dot_u))*np.sinc(f_phase*(1 - n_dot_u)))

    return(T_u_full)
    

def R_timing_michelson(f, n_hat, u, v, L, mode = "small antenna limit"):
    t_u = T_u (f, n_hat, u, L, mode)
    t_v = T_u (f, n_hat, v, L, mode)
    
    #uu = np.outer(u,u)
    #vv = np.outer(v,v)
    #R_timing = 0.5*(uu*t_u - vv*t_v)
    
    uu_tu = np.array([np.array([u[0]*u[0]*t_u, u[0]*u[1]*t_u, u[0]*u[2]*t_u]),\
                  np.array([u[1]*u[0]*t_u, u[1]*u[1]*t_u, u[1]*u[2]*t_u]),\
                  np.array([u[2]*u[0]*t_u, u[2]*u[1]*t_u, u[2]*u[2]*t_u])])
    
    vv_tv = np.array([np.array([v[0]*v[0]*t_v, v[0]*v[1]*t_v, v[0]*v[2]*t_v]),\
                  np.array([v[1]*v[0]*t_v, v[1]*v[1]*t_v, v[1]*v[2]*t_v]),\
                  np.array([v[2]*v[0]*t_v, v[2]*v[1]*t_v, v[2]*v[2]*t_v])])
    R_timing = 0.5*(uu_tu - vv_tv)
    
    return R_timing

def R_timing_michelson_small(f, n_hat, u, v, L):
    t_u = T_u_small (f, n_hat, u, L)
    t_v = T_u_small (f, n_hat, v, L)
    uu = np.outer(u,u)
    vv = np.outer(v,v)
    R_timing = 0.5*(uu*t_u - vv*t_v)
    return R_timing

def R_timing_michelson_full(f, n_hat, u, v, L):
    t_u = T_u_full (f, n_hat, u, L)
    t_v = T_u_full (f, n_hat, v, L)
    
    #uu = np.outer(u,u)
    #vv = np.outer(v,v)
    #R_timing = 0.5*(uu*t_u - vv*t_v)
    
    uu_tu = np.array([np.array([u[0]*u[0]*t_u, u[0]*u[1]*t_u, u[0]*u[2]*t_u]),\
                  np.array([u[1]*u[0]*t_u, u[1]*u[1]*t_u, u[1]*u[2]*t_u]),\
                  np.array([u[2]*u[0]*t_u, u[2]*u[1]*t_u, u[2]*u[2]*t_u])])
    
    #uu_tu = np.einsum('i,j,...',u,u,t_u)
    
    vv_tv = np.array([np.array([v[0]*v[0]*t_v, v[0]*v[1]*t_v, v[0]*v[2]*t_v]),\
                  np.array([v[1]*v[0]*t_v, v[1]*v[1]*t_v, v[1]*v[2]*t_v]),\
                  np.array([v[2]*v[0]*t_v, v[2]*v[1]*t_v, v[2]*v[2]*t_v])])
    #vv_tv = np.einsum('i,j,...',v,v,t_v)
    
    R_timing = 0.5*(uu_tu - vv_tv)
    return R_timing

def R_strain_michelson(f, n_hat, u, v, L, mode = "small antenna limit"):
    R_strain = R_timing_michelson(f, n_hat, u, v, L, mode)/(2*L/c)
    return R_strain


def R_strain_michelson_small(f, n_hat, u, v, L):
    R_strain = R_timing_michelson_small(f, n_hat, u, v, L)/(2*L/c)
    return R_strain

def R_strain_michelson_full(f, n_hat, u, v, L):
    R_strain = R_timing_michelson_full(f, n_hat, u, v, L)/(2*L/c)
    return R_strain

def Antenna_patterns(f, theta, phi, u, v , L, mode = "small antenna limit"):
    #if theta, phi numbers
    try:
        n_h = [n_hat(th, ph) for ph in phi for th in theta]
        
        e_cross, e_plus =np.transpose([e_pol(th, ph) for ph in phi for th in theta], axes = (1,0,2,3))

        R_cross = np.array([np.tensordot(e_cr, R_strain_michelson(f, n_hat, u, v, L, mode)) for n_hat, e_cr in zip(n_h, e_cross)])

        R_plus = np.array([np.tensordot(e_pl, R_strain_michelson(f, n_hat, u, v, L, mode)) for n_hat, e_pl in zip(n_h, e_plus)])
        
    except:
        n_h = n_hat(theta, phi)
        e_cross, e_plus = e_pol(theta, phi)

        R_cross = np.tensordot(e_cross, R_strain_michelson(f, n_h, u, v, L, mode))

        R_plus = np.tensordot(e_plus, R_strain_michelson(f, n_h, u, v, L, mode))
        
    return (R_cross, R_plus)

def Antenna_patterns_small_old(f, theta, phi, u, v , L):
    n, m, l = unit_vectors(theta, phi)
        
    try:
        e_cross, e_plus = np.transpose(e_pol_hats(m, l), axes = (0,3,4,1,2))
        
    except ValueError:
        e_cross, e_plus = e_pol_hats(m, l)
    
    try:
        R_mich = R_strain_michelson_small(f, n, u, v, L)
        #print(np.shape(e_cross), np.shape(R_mich))
        R_cross = np.tensordot(e_cross, R_mich)

        R_plus = np.tensordot(e_plus, R_mich)
        
    except ValueError:
        e_cross, e_plus = np.transpose(e_pol_hats(m, l), axes = (0,3,1,2))
        R_mich = R_strain_michelson_small(f, n, u, v, L)
        R_cross = np.tensordot(e_cross, R_mich)
        R_plus = np.tensordot(e_plus, R_mich)
        #print(np.shape(e_cross), np.shape(R_mich), np.shape(R_cross))
    return (R_cross, R_plus)

def Antenna_patterns_small(f, theta, phi, u, v , L):
    n, m, l = unit_vectors(theta, phi)
    #print(np.shape(n))    
    
    e_cross, e_plus = e_pol_hats(m, l)
    #print(np.shape(e_cross))
    
    R_mich = R_strain_michelson_small(f, n, u, v, L)
    #print(np.shape(R_mich))
    
    R_cross = np.einsum('...ij,ji', e_cross, R_mich)
    R_plus = np.einsum('...ij,ji', e_plus, R_mich)
    
    return (R_cross, R_plus)

def Antenna_patterns_full_old(f, theta, phi, u, v , L):
    n, m, l = unit_vectors(theta, phi)
    
    try:
        e_cross, e_plus = np.transpose(e_pol_hats(m, l), axes = (0,3,4,1,2))
        
    except ValueError:
        e_cross, e_plus = e_pol_hats(m, l)
    
    try:
        R_mich = R_strain_michelson_full(f, n, u, v, L)
        #print(np.shape(e_cross), np.shape(R_mich))
        try:
            R_cross = np.transpose(np.tensordot(e_cross, R_mich), axes=(2,3,0,1))[0][0]
            R_plus = np.transpose(np.tensordot(e_plus, R_mich), axes=(2,3,0,1))[0][0]
        except ValueError:
            R_cross = np.tensordot(e_cross, R_mich)
            R_plus = np.tensordot(e_plus, R_mich)
     
    except ValueError:
        e_cross, e_plus = np.transpose(e_pol_hats(m, l), axes = (0,3,1,2))
        R_mich = R_strain_michelson_full(f, n, u, v, L)
        #print(np.shape(e_cross), np.shape(R_mich))
        #R_cross = np.transpose(np.tensordot(e_cross, R_mich, axes = ([1,2],[1,0])))[0]
        #R_plus = np.transpose(np.tensordot(e_plus, R_mich, axes = ([1,2],[1,0])))[0]
        R_cross = np.einsum('...ij,ji...',e_cross, R_mich)#, optimize='greedy')
        R_plus = np.einsum('...ij,ji...',e_plus, R_mich)#, optimize='greedy')
        #print(np.shape(e_cross), np.shape(R_mich), np.shape(R_cross))
        
    return (R_cross, R_plus)

def Antenna_patterns_full(f, theta, phi, u, v , L):
    n, m, l = unit_vectors(theta, phi)
    
    #print(np.shape(n))    
    
    e_cross, e_plus = e_pol_hats(m, l)
    #print(np.shape(e_cross))
    
    R_mich = R_strain_michelson_full(f, n, u, v, L)
    #print(np.shape(R_mich))
    
    R_cross = np.einsum('...ij,ji...', e_cross, R_mich)
    R_plus = np.einsum('...ij,ji...', e_plus, R_mich)
    
    return (R_cross, R_plus)

def Antenna_patterns_small_non_GR(f, theta, phi, u, v , L):
    n, m, l = unit_vectors(theta, phi)
    #print(np.shape(n))    
    
    e_plus, e_cross, e_vec_x, e_vec_y, e_scal_breath, e_scal_lon = e_pol_hats_non_GR(n, m, l)
    #print(np.shape(e_cross))
    
    R_mich = R_strain_michelson_small(f, n, u, v, L)
    #print(np.shape(R_mich))
    
    R_cross = np.einsum('...ij,ji', e_cross, R_mich)
    R_plus = np.einsum('...ij,ji', e_plus, R_mich)
    
    R_vec_x = np.einsum('...ij,ji', e_vec_x, R_mich)
    R_vec_y = np.einsum('...ij,ji', e_vec_y, R_mich)
    
    R_scal_breath = np.einsum('...ij,ji', e_scal_breath, R_mich)
    R_scal_lon = np.einsum('...ij,ji', e_scal_lon, R_mich)
    
    
    return (R_cross, R_plus, R_vec_x, R_vec_y, R_scal_breath, R_scal_lon)


def Antenna_patterns_full_non_GR(f, theta, phi, u, v , L):
    n, m, l = unit_vectors(theta, phi)
    #print(np.shape(n))    
    
    e_plus, e_cross, e_vec_x, e_vec_y, e_scal_breath, e_scal_lon = e_pol_hats_non_GR(n, m, l)
    #print(np.shape(e_cross))
    
    R_mich = R_strain_michelson_full(f, n, u, v, L)
    #print(np.shape(R_mich))
    
    R_cross = np.einsum('...ij,ji...', e_cross, R_mich)
    R_plus = np.einsum('...ij,ji...', e_plus, R_mich)
    
    R_vec_x = np.einsum('...ij,ji...', e_vec_x, R_mich)
    R_vec_y = np.einsum('...ij,ji...', e_vec_y, R_mich)
    
    R_scal_breath = np.einsum('...ij,ji...', e_scal_breath, R_mich)
    R_scal_lon = np.einsum('...ij,ji...', e_scal_lon, R_mich)
    
    
    return (R_cross, R_plus, R_vec_x, R_vec_y, R_scal_breath, R_scal_lon)

####################
#ORFs               --> Remove old functions!!!!
####################
def ORF (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]),mode = "small antenna limit", beta = np.pi/2):
    #print(f)
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns(f, theta, phi, u1, v1 , L1, mode)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns(f, theta, phi, u2, v2 , L2, mode)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.sin(theta)*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi), (x1 - x2))/c))
        return geom_fact
    return (scipy.integrate.nquad(integrand,[[0, np.pi],[0, 2*np.pi]])[0]*(5/(8*np.pi*np.sin(beta)**2)))

def ORF_small (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2):
    #print(f)
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns_small(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns_small(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.sin(theta)*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi), (x1 - x2))/c))
        return geom_fact
    return (scipy.integrate.nquad(integrand,[[0, np.pi],[0, 2*np.pi]])[0]*(5/(8*np.pi*np.sin(beta)**2)))

def ORF_full (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2):
    #print(f)
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns_full(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns_full(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.sin(theta)*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi), (x1 - x2))/c))
        return geom_fact
    return (scipy.integrate.nquad(integrand,[[0, np.pi],[0, 2*np.pi]])[0]*(5/(8*np.pi*np.sin(beta)**2)))


def ORF_num(f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]),mode = "small antenna limit", beta = np.pi/2):
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns(f, theta, phi, u1, v1 , L1, mode)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns(f, theta, phi, u2, v2 , L2, mode)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.sin(theta)*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi), (x1 - x2))/c))
        return geom_fact
    
    theta = np.linspace(0, np.pi,50, endpoint=False)
    phi = np.linspace(0, 2*np.pi,100, endpoint=False)
    factor=  [integrand(th, ph)*5/(8*np.pi*np.sin(beta)**2) for ph in phi for th in theta]
    return np.real(np.sum(factor)*(2*np.pi/len(phi))*(np.pi/len(theta)))

def ORF_small_num (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2):
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns_small(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns_small(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.sin(theta)*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        return geom_fact
    theta = np.linspace(0, np.pi,100, endpoint=False)
    phi = np.linspace(0, 2*np.pi,200, endpoint=False)
    d_Omega = 2*np.pi/len(theta)*np.pi/len(phi)
    #factor=  [integrand(th, ph)*5/(8*np.pi*np.sin(beta)**2) for ph in phi for th in theta]
    
    theta, phi = np.meshgrid(theta, phi)
    factor = integrand(theta, phi)*5/(8*np.pi*np.sin(beta)**2)
    #print(np.real(np.sum(factor))*d_Omega)
    return np.real(np.sum(factor))*d_Omega

def ORF_small_num_healpy (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2, nside = 16, directional=False):
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns_small(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns_small(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        return geom_fact
    import healpy as hp
    phi, theta = hp.pix2ang(nside, range(0, int(12*nside**2)), lonlat=True)
    #print(np.max(theta), np.max(phi))
    theta = np.deg2rad(theta+90)
    phi = np.deg2rad(phi)
    
    d_Omega = 4*np.pi/(12*nside**2)
    
    #factor =  np.array([integrand(th, ph) for th, ph in zip(theta, phi)])*5/(8*np.pi*np.sin(beta)**2)
    if directional:
        orf_aniso = integrand(theta, phi)/2
        factor =  orf_aniso*5/(4*np.pi*np.sin(beta)**2)
        return np.real(np.sum(factor))*d_Omega, orf_aniso
    else:
        factor =  integrand(theta, phi)*5/(8*np.pi*np.sin(beta)**2)
        return np.real(np.sum(factor))*d_Omega

def ORF_full_num (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2):
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns_full(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns_full(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.sin(theta)*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        return geom_fact
    theta = np.linspace(0, np.pi,50, endpoint=False)
    phi = np.linspace(0, 2*np.pi,100, endpoint=False)
    d_Omega = 2*np.pi/len(theta)*np.pi/len(phi)
    #factor=  [integrand(th, ph)*5/(8*np.pi*np.sin(beta)**2) for ph in phi for th in theta]
    
    theta, phi = np.meshgrid(theta, phi)
    factor = integrand(theta, phi)*5/(8*np.pi*np.sin(beta)**2)
   
    return np.real(np.sum(factor))*d_Omega

def ORF_full_num_healpy (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2, nside = 16, directional = False):
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns_full(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns_full(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        return geom_fact
    import healpy as hp
    phi, theta = hp.pix2ang(nside, range(0, int(12*nside**2)), lonlat=True)
    #print(np.max(theta), np.max(phi))
    theta = np.deg2rad(theta+90)
    phi = np.deg2rad(phi)
    
    d_Omega = 4*np.pi/(12*nside**2)
    #factor =  np.array([integrand(th, ph) for th, ph in zip(theta, phi)])*5/(8*np.pi*np.sin(beta)**2)
    #factor = integrand(theta, phi)*5/(8*np.pi*np.sin(beta)**2)
   
    #return np.real(np.sum(factor))*d_Omega
    if directional:
        orf_aniso = integrand(theta, phi)/2
        factor =  orf_aniso*5/(4*np.pi*np.sin(beta)**2)
        return np.real(np.sum(factor))*d_Omega, orf_aniso
    else:
        factor =  integrand(theta, phi)*5/(8*np.pi*np.sin(beta)**2)
        return np.real(np.sum(factor))*d_Omega

def ORF_small_num_healpy_non_GR (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2, nside = 16):
    def integrand(theta, phi):
        R1_c, R1_p, R1_x, R1_y, R1_b, R1_l = Antenna_patterns_small_non_GR(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p, R2_x, R2_y, R2_b, R2_l = Antenna_patterns_small_non_GR(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_tensor = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        geom_vector = (R1_x*np.conjugate(R2_x) + R1_y*np.conjugate(R2_y))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        geom_scalar = (R1_b*np.conjugate(R2_b) + R1_l*np.conjugate(R2_l))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        return geom_tensor, geom_vector, geom_scalar
    
    import healpy as hp
    phi, theta = hp.pix2ang(nside, range(0, int(12*nside**2)), lonlat=True)
    #print(np.max(theta), np.max(phi))
    theta = np.deg2rad(theta+90)
    phi = np.deg2rad(phi)
    
    d_Omega = 4*np.pi/(12*nside**2)
    
    #factor =  np.array([integrand(th, ph) for th, ph in zip(theta, phi)])*5/(8*np.pi*np.sin(beta)**2)
    geom_tensor, geom_vector, geom_scalar =  integrand(theta, phi)
    
    factor_tensor = geom_tensor*5/(8*np.pi*np.sin(beta)**2)
    factor_vector = geom_vector*5/(8*np.pi*np.sin(beta)**2)
    factor_scalar = 2*geom_scalar*5/(8*np.pi*np.sin(beta)**2)
    
    #factor_tensor, factor_vector, factor_scalar =  integrand(theta, phi)*5/(8*np.pi*np.sin(beta)**2)
    
    orf_tensor = np.real(np.sum(factor_tensor))*d_Omega
    orf_vector = np.real(np.sum(factor_vector))*d_Omega
    orf_scalar = np.real(np.sum(factor_scalar))*d_Omega
    
    return orf_tensor, orf_vector, orf_scalar
    

def ORF_full_num_healpy_non_GR (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2, nside = 16):
    def integrand(theta, phi):
        R1_c, R1_p, R1_x, R1_y, R1_b, R1_l = Antenna_patterns_full_non_GR(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p, R2_x, R2_y, R2_b, R2_l = Antenna_patterns_full_non_GR(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_tensor = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        geom_vector = (R1_x*np.conjugate(R2_x) + R1_y*np.conjugate(R2_y))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        geom_scalar = (R1_b*np.conjugate(R2_b) + R1_l*np.conjugate(R2_l))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        return geom_tensor, geom_vector, geom_scalar
    
    import healpy as hp
    phi, theta = hp.pix2ang(nside, range(0, int(12*nside**2)), lonlat=True)
    #print(np.max(theta), np.max(phi))
    theta = np.deg2rad(theta+90)
    phi = np.deg2rad(phi)
    
    d_Omega = 4*np.pi/(12*nside**2)
    
    #factor =  np.array([integrand(th, ph) for th, ph in zip(theta, phi)])*5/(8*np.pi*np.sin(beta)**2)
    geom_tensor, geom_vector, geom_scalar =  integrand(theta, phi)
    
    factor_tensor = geom_tensor*5/(8*np.pi*np.sin(beta)**2)
    factor_vector = geom_vector*5/(8*np.pi*np.sin(beta)**2)
    factor_scalar = 2*geom_scalar*5/(8*np.pi*np.sin(beta)**2)
    
    #factor_tensor, factor_vector, factor_scalar =  integrand(theta, phi)*5/(8*np.pi*np.sin(beta)**2)
    
    orf_tensor = np.real(np.sum(factor_tensor))*d_Omega
    orf_vector = np.real(np.sum(factor_vector))*d_Omega
    orf_scalar = np.real(np.sum(factor_scalar))*d_Omega
    
    return orf_tensor, orf_vector, orf_scalar


def ORF_small_num_healpy_parity_violation (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2, nside = 16, directional=False):
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns_small(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns_small(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact_I = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        geom_fact_V = -1j*(R1_c*np.conjugate(R2_p) - R1_p*np.conjugate(R2_c))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        
        return geom_fact_I, geom_fact_V
    import healpy as hp
    phi, theta = hp.pix2ang(nside, range(0, int(12*nside**2)), lonlat=True)
    #print(np.max(theta), np.max(phi))
    theta = np.deg2rad(theta+90)
    phi = np.deg2rad(phi)
    
    d_Omega = 4*np.pi/(12*nside**2)
    
    #factor =  np.array([integrand(th, ph) for th, ph in zip(theta, phi)])*5/(8*np.pi*np.sin(beta)**2)
    if directional:
        orf_aniso_I, orf_aniso_V = integrand(theta, phi)
        factor_I =  orf_aniso_I*5/(8*np.pi*np.sin(beta)**2)
        factor_V =  orf_aniso_V*5/(8*np.pi*np.sin(beta)**2)
        return np.real(np.sum(factor_I))*d_Omega, orf_aniso_I/2, np.real(np.sum(factor_V))*d_Omega, orf_aniso_V/2
    else:
        factor_I, factor_V =  integrand(theta, phi)
        ORF_I = np.real(np.sum(factor_I))*d_Omega*5/(8*np.pi*np.sin(beta)**2)
        ORF_V = np.real(np.sum(factor_V))*d_Omega*5/(8*np.pi*np.sin(beta)**2)
        return ORF_I, ORF_V
    
def ORF_full_num_healpy_parity_violation (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2, nside = 16, directional=False):
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns_full(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns_full(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact_I = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        geom_fact_V = -1j*(R1_c*np.conjugate(R2_p) - R1_p*np.conjugate(R2_c))*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1 - x2)).T/c))
        
        return geom_fact_I, geom_fact_V
    import healpy as hp
    phi, theta = hp.pix2ang(nside, range(0, int(12*nside**2)), lonlat=True)
    #print(np.max(theta), np.max(phi))
    theta = np.deg2rad(theta+90)
    phi = np.deg2rad(phi)
    
    d_Omega = 4*np.pi/(12*nside**2)
    
    #factor =  np.array([integrand(th, ph) for th, ph in zip(theta, phi)])*5/(8*np.pi*np.sin(beta)**2)
    if directional:
        orf_aniso_I, orf_aniso_V = integrand(theta, phi)
        factor_I =  orf_aniso_I*5/(8*np.pi*np.sin(beta)**2)
        factor_V =  orf_aniso_V*5/(8*np.pi*np.sin(beta)**2)
        return np.real(np.sum(factor_I))*d_Omega, orf_aniso_I/2, np.real(np.sum(factor_V))*d_Omega, orf_aniso_V/2
    else:
        factor_I, factor_V =  integrand(theta, phi)
        ORF_I = np.real(np.sum(factor_I))*d_Omega*5/(8*np.pi*np.sin(beta)**2)
        ORF_V = np.real(np.sum(factor_V))*d_Omega*5/(8*np.pi*np.sin(beta)**2)
        return ORF_I, ORF_V

##################
#gamma_lm numerical
##################
def orf_lm(orf_aniso, l, m, nside):
    """
    Is the +90 (added for compatibility with orf_aniso) actually necessary in my whole package? 
    """
    if len(orf_aniso) != hp.nside2npix(nside):
        raise Exception("Pixel number not matching nside!")
    if abs(m)>l:
        raise Exception("m must be less or equal to l!")
        
    solid_angle = 4*np.pi/len(orf_aniso)
    phi, theta = hp.pix2ang(nside, range(0, int(12*nside**2)), lonlat=True)
    gamma_lm = np.sum(orf_aniso*sph_harm(m, l, np.deg2rad(phi), np.deg2rad(theta+90)))*solid_angle
    
    return gamma_lm

def orf_lm_t(orf_aniso, l, m, nside, t=0, t_mod = day_s):
        
    return orf_lm(orf_aniso, l, m, nside)*np.exp(-2j*np.pi*m*t/t_mod)


##################
#ISO ORFs - Bessel
##################

def j0(z):
    return scipy.special.spherical_jn(0,z)

def j1(z):
    return scipy.special.spherical_jn(1,z)

def j2(z):
    return scipy.special.spherical_jn(2,z)

def ORF_small_Bessel (f, u1, v1, u2, v2, L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2):
    
    R1 = R_strain_michelson_small(f, None, u1, v1, L1)
    R2 = R_strain_michelson_small(f, None, u2, v2, L2)
    alpha = 2*np.pi*f*np.linalg.norm(x2-x1)/c
    s_hat = (x2-x1)/np.linalg.norm(x2-x1)
    
    TrR1 = np.trace(R1)
    TrR2 = np.trace(R2)
    A_arg = TrR1*TrR2
    R1R2 = np.tensordot(R1, R2)
    B_arg = 2*R1R2
    ss = np.outer(s_hat,s_hat)
    C_arg = np.tensordot(TrR1*R2 + TrR2*R1, ss)
    D_arg = 4*np.tensordot(np.tensordot(np.transpose(R1),R2, axes = 1), ss)
    E_arg = np.tensordot(np.outer(R1,R2), np.outer(ss,ss))
    #print(A_arg, B_arg, C_arg, D_arg, E_arg, "\n")
    if alpha !=0:
        A_alpha = -2.5*j0(alpha) + (5/alpha)*j1(alpha) + (2.5/(alpha**2))*j2(alpha)
        B_alpha = 2.5*j0(alpha) - (5/alpha)*j1(alpha) + (2.5/(alpha**2))*j2(alpha)
        C_alpha = 2.5*j0(alpha) - (5/alpha)*j1(alpha) - (12.5/(alpha**2))*j2(alpha)
        D_alpha = -2.5*j0(alpha) + (10/alpha)*j1(alpha) - (12.5/(alpha**2))*j2(alpha)
        E_alpha = 2.5*j0(alpha) - (25/alpha)*j1(alpha) + (87.5/(alpha**2))*j2(alpha)
    else:
        A_alpha = -2.5*j0(alpha) + 11./6
        B_alpha = 2.5*j0(alpha) - 1.5
        C_alpha = 2.5*j0(alpha) - 2.5
        D_alpha = -2.5*j0(alpha) + 2.5
        E_alpha = 2.5*j0(alpha) - 7.5/3

    Gamma_12 = A_alpha*A_arg + B_alpha*B_arg + C_alpha*C_arg + D_alpha*D_arg + E_alpha*E_arg 

    gamma_12 = Gamma_12/(np.sin(beta)**2)
    #print(alpha, A_alpha, B_alpha, C_alpha, D_alpha, E_alpha)

    return gamma_12

def ORF_small_Bessel_vectorized (f, u1, v1, u2, v2, L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2):
    """
    Assumes the frequency to be an array
    """
    R1 = R_strain_michelson_small(f, None, u1, v1, L1)
    R2 = R_strain_michelson_small(f, None, u2, v2, L2)
    alpha = 2*np.pi*f*np.linalg.norm(x2-x1)/c
    s_hat = (x2-x1)/np.linalg.norm(x2-x1)
    
    TrR1 = np.trace(R1)
    TrR2 = np.trace(R2)
    A_arg = TrR1*TrR2
    R1R2 = np.tensordot(R1, R2)
    B_arg = 2*R1R2
    ss = np.outer(s_hat,s_hat)
    C_arg = np.tensordot(TrR1*R2 + TrR2*R1, ss)
    D_arg = 4*np.tensordot(np.tensordot(np.transpose(R1),R2, axes = 1), ss)
    E_arg = np.tensordot(np.outer(R1,R2), np.outer(ss,ss))
    #print(A_arg, B_arg, C_arg, D_arg, E_arg, "\n")
    
    A_alpha = np.where(alpha!=0., -2.5*j0(alpha) + (5/alpha)*j1(alpha) + (2.5/(alpha**2))*j2(alpha),\
                       -2./3)
    B_alpha = np.where(alpha!=0., 2.5*j0(alpha) - (5/alpha)*j1(alpha) + (2.5/(alpha**2))*j2(alpha),\
                       1.)
    C_alpha = np.where(alpha!=0., 2.5*j0(alpha) - (5/alpha)*j1(alpha) - (12.5/(alpha**2))*j2(alpha),\
                       0.)
    D_alpha = np.where(alpha!=0., -2.5*j0(alpha) + (10/alpha)*j1(alpha) - (12.5/(alpha**2))*j2(alpha),\
                       0.)
    E_alpha = np.where(alpha!=0., 2.5*j0(alpha) - (25/alpha)*j1(alpha) + (87.5/(alpha**2))*j2(alpha),\
                       0.)
    
    Gamma_12 = A_alpha*A_arg + B_alpha*B_arg + C_alpha*C_arg + D_alpha*D_arg + E_alpha*E_arg 

    gamma_12 = Gamma_12/(np.sin(beta)**2)
    #print(alpha, A_alpha, B_alpha, C_alpha, D_alpha, E_alpha)

    return gamma_12


############
#EARTH ROTATION --> REALLY NECESSARY FOR ISO ORFs??? NEEDs TO BE UPDATED TO HEALPIX!!!
############

def vector_earth_rotation(x0, t, t0 = 0):
    x_t = x0[0]*np.cos(Om_Earth*(t-t0)) - x0[1]*np.sin(Om_Earth*(t-t0))
    y_t = x0[0]*np.sin(Om_Earth*(t-t0)) + x0[1]*np.cos(Om_Earth*(t-t0))
    try:
        z_t = [x0[2]]*len(t)
    except TypeError:
        z_t = x0[2]
    #rot_matrix = [[np.cos(Om_Earth*(t-t0)), -np.sin(Om_Earth*(t-t0)), 0.]\
    #             [np.sin(Om_Earth*(t-t0)), np.cos(Om_Earth*(t-t0)), 0.]\
    #             [0., 0., 1.]]
    return np.array([x_t, y_t, z_t])

def ORF_small_time (f, u1, v1, u2, v2 , L1, L2, x1=np.array([0,0,0]), x2=np.array([0,0,0]), beta = np.pi/2, t = 0., t0 = 0.):
    def integrand(theta, phi):
        R1_c, R1_p = Antenna_patterns_small(f, theta, phi, u1, v1 , L1)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x1))
        R2_c, R2_p = Antenna_patterns_small(f, theta, phi, u2, v2 , L2)# * np.exp(2j*np.pi*(f/c)*np.dot(n_hat(theta, phi), x2))
        geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.sin(theta)*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, vector_earth_rotation(x1 - x2, t, t0)).T/c))
        #x1_rot = vector_earth_rotation(x1, t, t0)
        #x2_rot = vector_earth_rotation(x2, t, t0)
        #geom_fact = (R1_c*np.conjugate(R2_c) + R1_p*np.conjugate(R2_p))*np.sin(theta)*np.exp(2j*np.pi*f*(np.dot(n_hat(theta, phi).T, (x1_rot - x2_rot)).T/c))
        return geom_fact
    theta = np.linspace(0, np.pi,100, endpoint=False)
    phi = np.linspace(0, 2*np.pi,200, endpoint=False)
    d_Omega = 2*np.pi/len(theta)*np.pi/len(phi)
    #factor=  [integrand(th, ph)*5/(8*np.pi*np.sin(beta)**2) for ph in phi for th in theta]
    
    theta, phi = np.meshgrid(theta, phi)
    factor = integrand(theta, phi)*5/(8*np.pi*np.sin(beta)**2)
    #print(np.shape(factor), np.shape(np.sum(factor, axis=(1,2))))
    try:
        return np.real(np.sum(factor, axis=(1,2)))*d_Omega
    except:
        return np.real(np.sum(factor))*d_Omega
        
#######################
#DETECTOR POSITIONS FROM COORDINATES
#See Albert Lazzerini technical document, or Appendix B (shorter and clearer) from https://arxiv.org/pdf/gr-qc/0008066.pdf.
#######################

def detector_position_from_coordinates(lon=0., lat=0., elevation=0., azimuth_x=.0, azimuth_y=np.pi/2, tilt_x=0., tilt_y=0., beta = np.pi/2,radians = True):
    """
    It evalate detector and arms positions, given longitude [0,2*np.pi[->lambda
    latitude([-np.pi/2, np.pi/2])->phi, elevation (at most Mt.Everest height in meters),
    North from East azimuth_x (rad, 0 means looking at East of tilt = 0),
    North from East azimuth_y=azimuth_x+beta,
    tilt_x, tilt_y [0, 2*np.pi[ (angles between arm an tangent plane to central station),
    beta [0, 2*np.pi] (angle between detector arms)
    If radians is False (future) everything in degree (no angular minutes and seconds!) with longitude East and latitude North,
    to be converted in radians with the other angular quantities.
    The WGS-84 Earth model is assumed here.
    """
    #Semimajor axis in meters
    a = 6.378137e6
    b = 6.356752314e6
    
    def R_Earth(lat):
        return a**2/np.sqrt(a**2*np.cos(lat)**2 + b**2*np.sin(lat)**2)
    
    def detector_position(lon, lat, elevation):
        """
        Vector pointing from the center of Earth to
        the detector central station position.
        """
        R_phi = R_Earth(lat)
        #print(R_phi)
        x_d = (R_phi + elevation)*np.cos(lat)*np.cos(lon)
        y_d = (R_phi + elevation)*np.cos(lat)*np.sin(lon)
        z_d = ((b/a)**2*R_phi + elevation)*np.sin(lat)
        r_d = np.array([x_d, y_d, z_d])
        return r_d
    
    def unit_vectors_E_N_out(lon, lat, elevation):
        e_lon = np.array([-np.sin(lon), np.cos(lon), 0])
        e_lat = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
        e_elevation = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        
        return e_lon, e_lat, e_elevation
    
    def detector_arm(e_lon, e_lat, e_elevation, azimuth, tilt):
        """
        Unit vector of a detector arm, given central station position coordinates
        North from East azimuth and tilt.
        """
        arm_vector = np.array(np.cos(tilt)*np.cos(azimuth)*e_lon +\
                             np.cos(tilt)*np.sin(azimuth)*e_lat+\
                             np.sin(tilt)*e_elevation)
        return arm_vector
        
    if radians:
        x_D = detector_position(lon, lat, elevation)
        e_lon, e_lat, e_elevation = unit_vectors_E_N_out(lon, lat, elevation)
        u = detector_arm(e_lon, e_lat, e_elevation, azimuth_x, tilt_x)
        v = detector_arm(e_lon, e_lat, e_elevation, azimuth_y, tilt_y)
        #print(x_D, u, v)
        return x_D, u, v       
    else:
        print("Only radians for the time being")
        return
