import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.close("all")



# Angulos de pitch y roll saturados de manera de no pasar por singularidades. 

# Values used in Quadrotor_model_2020.slx
# Eduardo 23/01/23

m=0.9
g=9.81

# Ixx = .01 # Moment of Inertia
# Iyy = .01 # Moment of Inertia
# Izz = .03 # Moment of Inertia

# Input Saturation

umax=10
umin=-10



# External Trackings Gains 

k_v_x=2
k_p_x=1.6

k_v_y=2
k_p_y=3

# Internal Stability Gains

k_v_z=5
k_p_z=40 

k_v_psi=4
k_p_psi=7

k_v_theta=8
k_p_theta=120

k_v_phi=8
k_p_phi=120 

Pw=np.array([k_v_phi,k_v_theta,k_v_psi])
Pw=Pw.reshape(3,1)

Pq=-np.array([k_p_phi,k_p_theta,k_p_psi])
Pq=Pq.reshape(3,1)

#=========================================
# Quad Rotorcraft Full Model Functions
#=========================================

def Quad_Model(x,u,tau_psi, tau_theta, tau_phi):
    # X=>[x,dx,y,dy,z,dz,psi,dpsi,theta,dtheta,phi,dphi]
    # Luis Rodolfo Garcia Carrillo
    # Quad Rotorcraft Control Vision Based Hovering and Navigation
    # Springer 2013
    # Eqs. 2.30 - 2.35 pp. 28

    dX=x*0
    
    dX[0] = x[1]
    dX[1] = u*(np.sin(x[10])*np.sin(x[6])+np.cos(x[10])*np.cos(x[6])*np.sin(x[8]))/m
    dX[2] = x[3]
    dX[3] = u*(np.cos(x[10])*np.sin(x[8])*np.sin(x[6])-np.cos(x[6])*np.sin(x[10]))/m
    dX[4] = x[5]
    dX[5] = u*(np.cos(x[8])*np.cos(x[10]))/m-g
    
    dX[6] = x[7]
    dX[7] = tau_psi[0]
    dX[8] = x[9]
    dX[9]= tau_theta[0]
    dX[10]= x[11]
    dX[11]= tau_phi[0]
    
    return dX

#=========================================
# Auxiliary Math Functions for Quaternions
#=========================================

def ToQuad(Angs):
    # From Euler Angles to Quaternions
    phi,theta,psi=Angs
    q0=np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    q1=np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    q2=np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    q3=np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
    return np.vstack((q0,q1,q2,q3))

def ToAngs(Quad):
    # From Quaternions to Euler Angles
    q0,q1,q2,q3=Quad
    phi=np.arctan2(2*(q0*q1+q2*q3),q0**2-q1**2-q2**2+q3**2)
    theta=np.arcsin(2*(q0*q2-q3*q1))
    psi=np.arctan2(2*(q0*q3+q1*q2),q0**2+q1**2-q2**2-q3**2)
    
    return np.vstack((phi,theta,psi))

def Kron(p,q):
    # Kronecker product of 2 R4 vectors
    p0,p1,p2,p3=p
    q0,q1,q2,q3=q
    
    return np.vstack((p0*q0-p1*q1-p2*q2-p3*q3,
                      p0*q1+p1*q0+p2*q3-p3*q2,
                      p0*q2-p1*q3+p2*q0+p3*q1,
                      p0*q3+p1*q2-p2*q1+p3*q0))

def Conj(q):
    # Conjugate a R4 Quaternion
    q0,q1,q2,q3=q
    return  np.vstack((q0,-q1,-q2,-q3))

def Sat(u):
    # Saturation function
    u=u.flatten()
    uma=u*0+umax
    umi=u*0+umin
    umuum=np.vstack((uma,u,umi))
    
    aux=np.sort(umuum, axis=0)
    return aux[1]

#=========================================
# Quaternion Internal Control Function
#=========================================

def Tau(q,w,qref):
    q=q.reshape(4,1)
    w=w.reshape(3,1)
       
    qerr=Kron(qref,Conj(q))
    if q[0]<0:
        qerr=Conj(qerr)
    ttau=-Pq*qerr[1:4]-Pw*w
    #return Sat(ttau)
    return ttau

#==============================================
# Classic Internal Stability Control Functions
#==============================================

def Quad_tau_psi(dpsi,psi,psi_d):
    return -k_v_psi*dpsi-k_p_psi*(psi-psi_d)

def Quad_tau_theta(dtheta,theta,theta_d):
    return -k_v_theta*dtheta-k_p_theta*(theta-theta_d)

def Quad_tau_phi(dphi,phi,phi_d):
    return -k_v_phi*dphi-k_p_phi*(phi-phi_d)


#==============================================
# Tracking control Functions
#==============================================
 
def Quad_u(dz,z,theta,phi,z_d):
   return (-k_v_z*dz-k_p_z*(z-z_d)+m*g)/(np.cos(theta)*np.cos(phi))

def Quad_u2(z,dz,z_d,dz_d):
   return -k_v_z*(dz-dz_d)-k_p_z*(z-z_d)+m*g


def Quad_psi_d():
    return 0

def Quad_theta_d(x,x_d,dx,derivx_d):
    return np.arctan2(-(k_p_x*(x-x_d)+k_v_x*(dx-derivx_d)),g)

def Quad_phi_d(y,y_d,dy,derivy_d,theta):
    return np.arctan2((k_p_y*(y-y_d)+k_v_y*(dy-derivy_d))*np.cos(theta),g)

def funz_d(t):
    Z_d=t*0
    if t<0:
        Z_d=0 
    elif t<10:
        Z_d=t/10
    elif t<20:
        Z_d=1
    elif t<30:
        Z_d=t*(0.5-1)/(30-20)+(1-20*(0.5-1)/(30-20))
    else:
        Z_d=0.5
    return Z_d

# def funx_d(t):
#     if t<0:
#         X_d=0 
#     elif t<4:
#         X_d=t/4 
#     else:
#         X_d=1
#     return X_d     

# def funy_d(t):
#     if t<0:
#         Y_d=0 
#     elif t<4:
#         Y_d=t/4 
#     else:
#         Y_d=1
#     return Y_d   

def funx_d(t):
    f=.1/(2*np.pi)
    return np.sin(2*np.pi*f*t)

def funderivx_d(t):
    f=.1/(2*np.pi)
    return np.cos(2*np.pi*f*t)*2*np.pi*f

def funy_d(t):
    f=.1/(2*np.pi)
    return np.sin(2*np.pi*f*t)*0

def funderivy_d(t):
    f=.1/(2*np.pi)
    return np.cos(2*np.pi*f*t)*2*np.pi*f*0

#==============================================
# Classic Closed-Loop Integration Functions
#==============================================

def Quad_Inte(t,X):
    x,dx,y,dy,z,dz,psi,dpsi,theta,dtheta,phi,dphi=X
    
    x_d=funx_d(t)
    derivx_d=funderivx_d(t)
    y_d=funy_d(t)
    derivy_d=funderivy_d(t)
    z_d=funz_d(t)
    
    phi_d=Quad_phi_d(y,y_d,dy,derivy_d,theta)
    theta_d=Quad_theta_d(x,x_d,dx,derivx_d)
    psi_d=Quad_psi_d()
    
    # u=Quad_u(dz,z,theta,phi,z_d)
    u=Quad_u2(z,dz,z_d,0)
    
    tau_psi=Quad_tau_psi(dpsi,psi,psi_d)
    tau_theta=Quad_tau_theta(dtheta,theta,theta_d)
    tau_phi=Quad_tau_phi(dphi,phi,phi_d)
       
    return Quad_Model(X,u,tau_psi, tau_theta, tau_phi)

#==============================================
# Quaternion Closed-Loop Integration Functions
#==============================================

def refphi(t):
    if t<1+40:
        phi_d=0
    elif t<2+40:
        phi_d=(t-40)*(-np.pi*1.1-0)/(2-1)+np.pi*1.1
    elif t<3+40:  
        phi_d=(t-40)*(-np.pi*1.1-0)/(3-2)+3*np.pi*1.1
    else:
        phi_d=0
    return phi_d 

def Quad_Inte_Quaternions(t,X):
    x,dx,y,dy,z,dz,psi,dpsi,theta,dtheta,phi,dphi=X
    
    x_d=funx_d(t)
    derivx_d=funderivx_d(t)
    y_d=funy_d(t)
    derivy_d=funderivy_d(t)
    z_d=funz_d(t)
        
    phi_d=Quad_phi_d(y,y_d,dy,derivy_d,theta)
    theta_d=Quad_theta_d(x,x_d,dx,derivx_d)
    psi_d=Quad_psi_d()
        
    phi_d=refphi(t)
  
    u=Quad_u(dz,z,theta,phi,z_d)
    
    q=ToQuad(np.array([phi,theta,psi]))
    qref=ToQuad(np.array([phi_d,theta_d,psi_d]))
    w=np.array([dphi,dtheta,dpsi])
    
    tau_phi,tau_theta,tau_psi=Tau(q,w,qref)
      
    return Quad_Model(X,u,tau_psi, tau_theta, tau_phi)
      
#========================================



XI=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
t_span=[0,75]

# sol=sp.integrate.solve_ivp(Quad_Inte, t_span, XI)
sol=sp.integrate.solve_ivp(Quad_Inte_Quaternions, t_span, XI)
# sol=sp.integrate.solve_ivp(Quad_Inte_Quaternions, t_span, XI, method='LSODA', max_step=1e-3,min_step=1e-3,first_step=1e-3)


myt=sol.t

myx=sol.y[0,:]
mydx=sol.y[1,:]
myy=sol.y[2,:]
mydy=sol.y[3,:]
myz=sol.y[4,:]
mydz=sol.y[5,:]
mypsi=sol.y[6,:]
mydpsi=sol.y[7,:]
mytheta=sol.y[8,:]
mydtheta=sol.y[9,:]
myphi=sol.y[10,:]
mydphi=sol.y[11,:]

myu=myt*0
tau_phi=myt*0
tau_theta=myt*0
tau_psi=myt*0

myrefphi=myt*0
myreftheta=myt*0
myrefpsi=myt*0
U=myt*0

q=np.zeros((4,len(myt)))
for i in np.arange(len(myt)):
       
    q[:,i]=ToQuad(np.array([myphi[i],mytheta[i],mypsi[i]])).flatten()
    qref=ToQuad(np.array([0,0,0]))
    w=np.array([mydphi[i],mydtheta[i],mydpsi[i]])
    
    # myu[i]=Quad_u(mydz[i],myz[i],mytheta[i],myphi[i],funz_d(myt[i]))
    myu[i]=Quad_u2(myz[i],mydz[i],funz_d(myt[i]),0)
    tau_phi[[i]],tau_theta[[i]],tau_psi[[i]]=Tau(q[:,i],w,qref)
    
    myrefphi[i]=refphi(myt[i])
    myreftheta[i]=Quad_theta_d(myx[i],funx_d(myt[i]),mydx[i],funderivx_d(myt[i]))
    myrefpsi[i]=0
    
    U[i]=Quad_u2(myz[i],mydz[i],funz_d(myt[i]),0)
    
    if np.abs(myphi[i])>np.pi:
        myphi[i]=myphi[i]-np.sign(myphi[i])*2*np.pi
    if np.abs(mytheta[i])>np.pi:
        mytheta[i]=mytheta[i]-np.sign(mytheta[i])*2*np.pi
    if np.abs(mypsi[i])>np.pi:
        mypsi[i]=mypsi[i]-np.sign(mypsi[i])*2*np.pi

plt.figure(0)
plt.plot(myt,myx)
plt.plot(myt,mydx)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('x vs dx')

plt.figure(1)
plt.plot(myt,myy)
plt.plot(myt,mydy)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('y vs dy')

plt.figure(2)
plt.plot(myt,myz)
plt.plot(myt,mydz)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('z vs dz')

plt.figure(3)
plt.plot(myt,mypsi)
plt.plot(myt,myrefpsi,'-k')
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('psi vs psi_d')

plt.figure(4)
plt.plot(myt,mytheta)
plt.plot(myt,myreftheta,'-k')
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('theta vs theta_d')

plt.figure(5)
plt.plot(myt,myphi)
plt.plot(myt,myrefphi,'-k')
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('phi vs phi_d')

# plt.figure(6)
# plt.plot(myt,funx_d(myt))
# plt.plot(myt,funderivx_d(myt))
# plt.minorticks_on()
# plt.grid(which='major')
# plt.grid(which='minor')
# plt.title('x_d vs dot x_d')

# plt.figure(7)
# plt.plot(myt,funy_d(myt))
# plt.plot(myt,funderivy_d(myt))
# plt.minorticks_on()
# plt.grid(which='major')
# plt.grid(which='minor')
# plt.title('y_d vs dot y_d')


plt.figure(30)
plt.plot(myt,U)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('U')

plt.figure(31)
plt.plot(myt,tau_phi)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('tau_phi')

plt.figure(41)
plt.plot(myt,tau_theta)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('tau_theta')

plt.figure(51)
plt.plot(myt,tau_psi)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('tau_psi')

plt.figure(6)
plt.plot(myt,q[0,:])
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor')
plt.title('q0')
    