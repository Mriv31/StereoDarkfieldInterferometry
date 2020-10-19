#!/usr/bin/env python
# coding: utf-8




import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle,Circle
from matplotlib import colors
import PyMieScatt as ps
import warnings
warnings.filterwarnings('ignore')



def scattered_intensity(ncore=1.6,nlayer=1.333,dcore=2800,dlayer=0,wavelength=650,slits=[[0.4,0],[1.1,0],[-0.4,0],[-1.1,0]],slitswidth=0.2,slitslength=0.7,medium=1.33,noil=1.51,xmax=3,ymax=2,focal=1.8,log=1,thetaind=23,plot=1):
    ncore = ncore/medium #complex refractive index of the scatterer compared to the refractive index of the medium
    wavelength = wavelength/medium #wavelength of the light in the medium (here water)
    #Use PyMieScatt to compute the MIE Scattering matrix
    if dlayer > 0:
        theta,SL,SR,SU  = ps.CoreShellScatteringFunction(ncore, nlayer, wavelength, dcore, dlayer, minAngle=0, maxAngle=60, angularResolution=0.5) #scattered intensity (L:parallel/R:perpendicular/U:unpolarized)
    else: 
        theta,SL,SR,SU  = ps.ScatteringFunction(ncore,wavelength,dcore,maxAngle=60,angularResolution=0.5,nMedium=1) #scattered intensity (L:parallel/R:perpendicular/U:unpolarized)
    thetai = thetaind/180*np.pi    
      
    SU = SU*wavelength**2*1e-18/(4*np.pi**2) # intensity of the scattered light at a distance 1 from the sphere for an incoming intensity of 1 W/m**2
    
    SU = np.sin(theta)*SU # intensity emitted between theta, theta+dtheta,  phi, phi+dphi for an incoming intensity of 1 W/m**2
    
    thetai = np.arcsin(np.sin(thetai)/medium) #incidende angle in the medium
    N = 1000 #discretization of phi
    phi = np.linspace(0,2*np.pi,N)
    dphi = phi[1]-phi[0] #dphi
    dtheta = theta[1]-theta[0] #dhetta
    tv,pv = np.meshgrid(theta,phi)
    Int=np.array([SU,])
    Int = np.repeat(Int, repeats=len(phi), axis=0)
    Int = Int*dtheta*dphi  #scattered intensity between theta+dtheta, dphi+dphi
    
    #Computation of the normed wave vectors corresponding to a scattering of theta, phi, from the incident angle 
    k_x = np.sin(tv)*np.cos(pv)*np.cos(thetai)+np.cos(tv)*np.sin(thetai) #coord_x_vector_angle_theta_phi_with_incident
    k_y = np.sin(tv)*np.sin(pv) #coord_y_vector_angle_theta_phi_with_incident
    k_z =  -np.sin(thetai)*np.sin(tv)*np.cos(pv)+np.cos(thetai)*np.cos(tv) #coord_z_vector_angle_theta_phi_with_incident
    
    #Correction of the wave vector due to the refraction at the planar interface medium/oil.
    k_z = np.sqrt(noil**2/medium**2-k_x**2-k_y**2)
    
    #Coordinates of the Fourier focal point conjugated with the rays of angles phi, theta
    x = k_x *focal / k_z * noil 
    y = k_y *focal / k_z * noil    
    
    #Computation of the jacobian for the transformation phi, theta -> x,y
    dxth = x - np.roll(x,1,axis=0)
    dxp  = x - np.roll(x,1,axis=1)
    dyth = y - np.roll(y,1,axis=0)
    dyp  =  y - np.roll(y,1,axis=1)
    jacobien = np.abs(dxth*dyp-dxp*dyth)
    Int2 = Int/jacobien #normalized intensity recieved in the fourier plane between x, x+dx, y+dy for an incoming intensity of 1 W/m**2


    x = x[1:-1,:].flatten()
    y = y[1:-1,:].flatten()
    Int2 = Int2[1:-1,:].flatten()
    tv = tv[1:-1,:].flatten()
    Int = Int[1-1:,:].flatten()


    ind = np.where(np.logical_and(np.abs(x)<xmax,np.abs(y)<ymax))[0] #only select a part of the Fourier plane
    x = x[ind]
    y = y[ind]
    Int = Int[ind]   
    Int2 = Int2[ind]
    tv=tv[ind]

    
    #plot the scattered intensity in the Fourier plane as well as the slits and focusing point of the incoming light
    if plot:
        plt.figure(figsize=(12,8))


        if (log):
            dif = plt.scatter(x[1:],y[1:],c=np.log10(Int2[1:]))
                            
        else:
            dif = plt.scatter(x[1:],y[1:],c=Int2[1:],s=4)

        fig=plt.gcf()
        fig.colorbar(dif)
        ax=plt.gca()
        ax.set_aspect("equal")
        for i in range(len(slits)):
            ax.add_patch(Rectangle((slits[i][0]-slitswidth/2,slits[i][1]-slitslength/2),slitswidth,slitslength,linewidth=1,edgecolor='r',facecolor='None'))
        ax.add_patch(Circle((np.tan(thetaind/180*np.pi)*focal,0),0.05,linewidth=1,edgecolor='r',facecolor='None'))
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.xlim([-xmax,xmax])
        plt.ylim([-ymax,ymax])
        plt.title("Scattered light intensity by one particle as observed in the Fourier plane. \n Colors indicate intensity per area (W/mm^2)")
        plt.show()
        
        
    print("Intensities of scattered light going through the slits for an incoming irradiance of 1W/m^2")    
    #Compute the intensity of scattered light going through each slits
    Intensity=[]    
    for i in range(len(slits)):
        Intensity.append(integrate_in_fringe(x,y,Int,slits[i][0],slits[i][1],slitswidth,slitslength))
        print("Slit with position ["+str(slits[i][0])+","+str(slits[i][1])+"] : " + str(Intensity[-1]) + "W")
        

    


    return x,y,Int


def integrate_in_fringe(kxl,kyl,Int,cx,cy,width,height,norm=None):
    xl = cx-width/2 
    yl = cy-height/2
    xu = cx+width/2 
    yu = cy+height/2
    ind = np.where(np.logical_and.reduce([kxl>xl,kxl<xu,kyl>yl,kyl<yu]))[0] 
    kxa = kxl[ind]
    kya = kyl[ind]
    Inta = Int[ind]
    return np.sum(Inta)
    
    

 

