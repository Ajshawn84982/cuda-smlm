# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:37:38 2023

@author: localadmin
"""


from tifffile import imread
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import pickle
import torch
from skimage.registration import  phase_cross_correlation
from spot_detection import spot_detection_method
from tqdm import tqdm

class GaussianPSF_biplane:
    def __init__(self,roi=21):
        self.theta=np.array([0,0,0,0,0,0]) #x,y,z,I,bg,sigma
        self.roi=roi
        self.pixelsize=100.0
        self.xx,self.yy=np.meshgrid(np.linspace(0,roi-1,roi),np.linspace(0,roi-1,roi))
        self.xx=(self.xx*self.pixelsize)
        self.yy=(self.yy*self.pixelsize)
        
        self.R=0.5
        self.R2=1.0-self.R

        
        self.sigma0ch1=124.14
        self.sigma0ch2=124.14
        self.sqrtsigma=np.sqrt(2.0)*self.sigma0ch1
        self.sqrtpi=np.sqrt(np.pi)
        self.halfpixel=0.5*self.pixelsize
        self.cuda0=torch.device('cuda:0')
        self.xxg= torch.tensor(self.xx,device=self.cuda0).to(torch.float32)
        self.yyg= torch.tensor(self.yy,device=self.cuda0).to(torch.float32)
    def Exyfun_gpu(self):
        
        l=len(self.theta)
        
        xx = self.xxg.unsqueeze(0).repeat(l, 1)
        yy = self.yyg.unsqueeze(0).repeat(l, 1)
        thetax=torch.tensor(self.theta[:,0],device=self.cuda0).unsqueeze(1).repeat(1, self.roi*self.roi)
        thetay=torch.tensor(self.theta[:,1],device=self.cuda0).unsqueeze(1).repeat(1, self.roi*self.roi)
        self.Ex=0.5*(torch.erf((xx-thetax+self.halfpixel)/(self.sqrtsigma))-torch.erf((self.xx-thetax-self.halfpixel)/(self.sqrtsigma)))
        self.Ey=0.5*(torch.erf((yy-thetay+self.halfpixel)/(self.sqrtsigma))-torch.erf((self.yy-thetay-self.halfpixel)/(self.sqrtsigma)))
    
    
    def ukfun_ch1_gpu(self):

        thetax=self.theta[:,0].clone().detach().to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        thetay=self.theta[:,1].clone().detach().to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        
        
        #thetax=torch.tensor(self.theta[:,0],device=self.cuda0).to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        #thetay=torch.tensor(self.theta[:,1],device=self.cuda0).to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        self.thetax=thetax
        self.thetay=thetay
        #thetax=torch.tensor(self.theta[:,0],device=self.cuda0).unsqueeze(1).repeat(1, roisq)
        #thetay=torch.tensor(self.theta[:,1],device=self.cuda0).unsqueeze(1).repeat(1, roisq)
        
        self.Ex=0.5*(torch.erf((self.xxgpu-thetax+self.halfpixel)/(self.sqrtsigma))-torch.erf((self.xxgpu-thetax-self.halfpixel)/(self.sqrtsigma)))
        self.Ey=0.5*(torch.erf((self.yygpu-thetay+self.halfpixel)/(self.sqrtsigma))-torch.erf((self.yygpu-thetay-self.halfpixel)/(self.sqrtsigma)))
        #self.I=torch.tensor(self.theta[:,2],device=self.cuda0).to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        self.I=self.theta[:,2].clone().detach().to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        bg=self.theta[:,3].clone().detach().to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        
    
        #bg=torch.tensor(self.theta[:,3],device=self.cuda0).to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        #I=torch.tensor(self.theta[:,2],device=self.cuda0).unsqueeze(1).repeat(1, roisq)
        #bg=torch.tensor(self.theta[:,3],device=self.cuda0).unsqueeze(1).repeat(1, roisq)
        self.dI_ch1=self.Ex*self.Ey
        self.uk_ch1=self.dI_ch1*self.I+bg
        torch.cuda.synchronize()
    
    def ukfun_sigma_gpu(self):
        #x,y,I,bg,sigma
        #self.sigmafun_ch1()
        #self.Exyfun_gpu()
        self.sigma0ch1=self.theta[:,4]
        self.sqrtsigma=np.sqrt(2.0)*self.sigma0ch1.unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        #xx = self.xxg.unsqueeze(0).repeat(l, 1, 1)
        #yy = self.yyg.unsqueeze(0).repeat(l, 1, 1)
        thetax=self.theta[:,0].clone().detach().to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        thetay=self.theta[:,1].clone().detach().to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        
        
        #thetax=torch.tensor(self.theta[:,0],device=self.cuda0).to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        #thetay=torch.tensor(self.theta[:,1],device=self.cuda0).to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        self.thetax=thetax
        self.thetay=thetay
        #thetax=torch.tensor(self.theta[:,0],device=self.cuda0).unsqueeze(1).repeat(1, roisq)
        #thetay=torch.tensor(self.theta[:,1],device=self.cuda0).unsqueeze(1).repeat(1, roisq)
        
        
        self.Ex=0.5*(torch.erf((self.xxgpu-thetax+self.halfpixel)/(self.sqrtsigma))-torch.erf((self.xxgpu-thetax-self.halfpixel)/(self.sqrtsigma)))
        self.Ey=0.5*(torch.erf((self.yygpu-thetay+self.halfpixel)/(self.sqrtsigma))-torch.erf((self.yygpu-thetay-self.halfpixel)/(self.sqrtsigma)))
        #self.I=torch.tensor(self.theta[:,2],device=self.cuda0).to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        self.I=self.theta[:,2].clone().detach().to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        bg=self.theta[:,3].clone().detach().to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        
    
        #bg=torch.tensor(self.theta[:,3],device=self.cuda0).to(torch.float32).unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi)
        #I=torch.tensor(self.theta[:,2],device=self.cuda0).unsqueeze(1).repeat(1, roisq)
        #bg=torch.tensor(self.theta[:,3],device=self.cuda0).unsqueeze(1).repeat(1, roisq)
        self.dI_ch1=self.Ex*self.Ey
        self.uk_ch1=self.dI_ch1*self.I+bg
        torch.cuda.synchronize()
    
    def dxysigmafun_gpu(self):
        
        self.sigma0ch1=torch.tensor(self.theta[:,4].unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi),device=self.cuda0)
        #self.sqrtsigma=torch.tensor(np.sqrt(2.0)*self.sigma0ch1,device=self.cuda0)
        l=len(self.theta)

        
        coef1=1.0/(np.sqrt(2.0*np.pi)*self.sigma0ch1)
        coef2=(2.0*self.sigma0ch1**2)
        x1=self.xxgpu-self.thetax-self.halfpixel
        x2=self.xxgpu-self.thetax+self.halfpixel
        y1=self.yygpu-self.thetay-self.halfpixel
        y2=self.yygpu-self.thetay+self.halfpixel
        self.dx_ch1=coef1*self.I*(torch.exp(-(x1)**2/coef2)-torch.exp(-(x2)**2/coef2))*self.Ey
        self.dy_ch1=coef1*self.I*(torch.exp(-(y1)**2/coef2)-torch.exp(-(y2)**2/coef2))*self.Ex
        self.dbg_ch1=torch.ones([l,self.roi,self.roi],device=self.cuda0)
        
        coef3=-(2.0/self.sqrtpi)*(1.0/(self.sigma0ch1**2))/(np.sqrt(2.0))
        self.dExdsigma=0.5*(coef3*(x2)*torch.exp(-(x2)**2/(2*self.sigma0ch1**2)))-0.5*(coef3*((x1))*torch.exp(-(x1)**2/(2.0*self.sigma0ch1**2)))
        self.dEydsigma=0.5*(coef3*(y2)*torch.exp(-(y2)**2/(2*self.sigma0ch1**2)))-0.5*(coef3*((y1))*torch.exp(-(y1)**2/(2.0*self.sigma0ch1**2)))
        self.dsigma=self.I*(self.dExdsigma*self.Ey+self.dEydsigma*self.Ex)
        
        torch.cuda.synchronize()
    
    
    def dxyfun_ch1_gpu(self):
        
        l=len(self.theta)

        
        coef1=self.R/(np.sqrt(2*np.pi)*self.sigma0ch1)
        coef2=(2.0*self.sigma0ch1**2)
        self.dx_ch1=coef1*self.I*(torch.exp(-(self.xxgpu-self.thetax-self.halfpixel)**2/coef2)-torch.exp(-(self.xxgpu-self.thetax+self.halfpixel)**2/coef2))*self.Ey
        self.dy_ch1=coef1*self.I*(torch.exp(-(self.yygpu-self.thetay-self.halfpixel)**2/coef2)-torch.exp(-(self.yygpu-self.thetay+self.halfpixel)**2/coef2))*self.Ex
        self.dbg_ch1=torch.ones([l,self.roi,self.roi],device=self.cuda0)
        torch.cuda.synchronize()
    
    
    
    def Exfun(self):
        self.Ex=0.5*(erf((self.xx-self.theta[0]+self.halfpixel)/(self.sqrtsigma))-erf((self.xx-self.theta[0]-self.halfpixel)/(self.sqrtsigma)))
        
    def Eyfun(self):
        self.Ey=0.5*(erf((self.yy-self.theta[1]+self.halfpixel)/(self.sqrtsigma))-erf((self.yy-self.theta[1]-self.halfpixel)/(self.sqrtsigma)))
                         
    def dExdsigmafun_ch1(self):
        self.dExdsigma_ch1=0.5*(-(2.0/self.sqrtpi)*(1.0/(self.sigma0ch1**2))*((self.xx-self.theta[0]+self.halfpixel)/(np.sqrt(2)))*np.exp(-(self.xx-self.theta[0]+self.halfpixel)**2/(2*self.sigma0ch1**2)))-0.5*(-(2.0/self.sqrtpi)*(1.0/(self.sigma0ch1**2))*((self.xx-self.theta[0]-self.halfpixel)/(np.sqrt(2.0)))*np.exp(-(self.xx-self.theta[0]-self.halfpixel)**2/(2.0*self.sigma0ch1**2)))
    
    
    def dEydsigmafun_ch1(self):
        self.dEydsigma_ch1=0.5*(-(2.0/self.sqrtpi)*(1.0/(self.sigma0ch1**2))*((self.yy-self.theta[1]+self.halfpixel)/(np.sqrt(2)))*np.exp(-(self.yy-self.theta[1]+self.halfpixel)**2/(2*self.sigma0ch1**2)))-0.5*(-(2.0/self.sqrtpi)*(1.0/(self.sigma0ch1**2))*((self.yy-self.theta[1]-self.halfpixel)/(np.sqrt(2.0)))*np.exp(-(self.yy-self.theta[1]-self.halfpixel)**2/(2.0*self.sigma0ch1**2)))                                                                  

    def dExdsigmafun_ch2(self):
        self.dExdsigma_ch2=0.5*(-(2.0/self.sqrtpi)*(1.0/(self.sigma0ch1**2))*((self.xx-self.theta[0]+self.halfpixel)/(np.sqrt(2)))*np.exp(-(self.xx-self.theta[0]+self.halfpixel)**2/(2*self.sigma0ch1**2)))-0.5*(-(2.0/self.sqrtpi)*(1.0/(self.sigma0ch1**2))*((self.xx-self.theta[0]-self.halfpixel)/(np.sqrt(2.0)))*np.exp(-(self.xx-self.theta[0]-self.halfpixel)**2/(2.0*self.sigma0ch1**2)))
                                                                                                                                                                                        
    def dEydsigmafun_ch2(self):
        self.dEydsigma_ch2=0.5*(-(2.0/self.sqrtpi)*(1.0/(self.sigma0ch1**2))*((self.yy-self.theta[1]+self.halfpixel)/(np.sqrt(2)))*np.exp(-(self.yy-self.theta[1]+self.halfpixel)**2/(2*self.sigma0ch1**2)))-0.5*(-(2.0/self.sqrtpi)*(1.0/(self.sigma0ch1**2))*((self.yy-self.theta[1]-self.halfpixel)/(np.sqrt(2.0)))*np.exp(-(self.yy-self.theta[1]-self.halfpixel)**2/(2.0*self.sigma0ch1**2)))                                                                                                                                                                              
    
    def ddudsigmafun_ch1(self):

        self.ddudsigma_ch1=self.theta[2]*(self.ddExdsigma_ch1*self.Ey+2.0*self.dExdsigma_ch1*self.dEydsigma_ch1+self.ddEydsigma_ch1)
    
    def ddudsigmafun_ch2(self):

        self.ddudsigma_ch2=self.theta[2]*(self.ddExdsigma_ch2*self.Ey+2.0*self.dExdsigma_ch2*self.dEydsigma_ch2+self.ddEydsigma_ch2)
    
    
    def ukfun_ch1_nolink(self):
        #self.sigmafun_ch1()
        self.Exfun()
        self.Eyfun()
        self.uk_ch1=self.R*self.theta[2]*self.Ex*self.Ey+self.theta[3]
        
    def ukfun_ch2_nolink(self):
        #self.sigmafun_ch2()
        self.Exfun()
        self.Eyfun()
        self.uk_ch2=self.R2*self.theta[2]*self.Ex*self.Ey+self.theta[5]
    
    
    def ukfun_ch1(self):
        #self.sigmafun_ch1()
        self.Exfun()
        self.Eyfun()
        self.uk_ch1=self.R*self.theta[2]*self.Ex*self.Ey+self.theta[3]*self.R
        
    def ukfun_ch2(self):
        #self.sigmafun_ch2()
        self.Exfun()
        self.Eyfun()
        self.uk_ch2=self.R2*self.theta[2]*self.Ex*self.Ey+self.theta[3]*self.R2
    
    def dxfun_ch1(self):
        
        self.dx_ch1=(self.theta[2]*self.R/(np.sqrt(2*np.pi)*self.sigma0ch1))*(np.exp(-(self.xx-self.theta[0]-self.halfpixel)**2/(2*self.sigma0ch1**2))-np.exp(-(self.xx-self.theta[0]+self.halfpixel)**2/(2*self.sigma0ch1**2)))*self.Ey
    
    def dyfun_ch1(self):
        
        self.dy_ch1=(self.theta[2]*self.R/(np.sqrt(2*np.pi)*self.sigma0ch1))*(np.exp(-(self.yy-self.theta[1]-self.halfpixel)**2/(2*self.sigma0ch1**2))-np.exp(-(self.yy-self.theta[1]+self.halfpixel)**2/(2*self.sigma0ch1**2)))*self.Ex
        
    def dxfun_ch2(self):
        
        self.dx_ch2=(self.theta[2]*self.R2/(np.sqrt(2*np.pi)*self.sigma0ch1))*(np.exp(-(self.xx-self.theta[0]-self.halfpixel)**2/(2*self.sigma0ch1**2))-np.exp(-(self.xx-self.theta[0]+self.halfpixel)**2/(2*self.sigma0ch1**2)))*self.Ey
    
    def dyfun_ch2(self):
        
        self.dy_ch2=(self.theta[2]*self.R2/(np.sqrt(2*np.pi)*self.sigma0ch1))*(np.exp(-(self.yy-self.theta[1]-self.halfpixel)**2/(2*self.sigma0ch1**2))-np.exp(-(self.yy-self.theta[1]+self.halfpixel)**2/(2*self.sigma0ch1**2)))*self.Ex
    
    def dIfun_ch1(self):
        self.dI_ch1=self.Ex*self.Ey*self.R
        
    def dIfun_ch2(self):
        self.dI_ch2=self.Ex*self.Ey*self.R2
    def dRfun_ch1(self):
        self.dR_ch1=self.theta[2]*self.Ex*self.Ey+self.theta[3]
        
    def dRfun_ch2(self):
        self.dR_ch2=-self.theta[2]*self.Ex*self.Ey-self.theta[3]
    
    def dbgfun_ch1_nolink(self):
        self.dbg_ch1=1.0
        
    def dbgfun_ch2_nolink(self):
        self.dbg_ch2=1.0
    
    def dbgfun_ch1(self):
        self.dbg_ch1=self.R
        
    def dbgfun_ch2(self):
        self.dbg_ch2=self.R2
    
    def ddIfun_ch1(self):
        self.ddI_ch1=0
    def ddIfun_ch2(self):
        self.ddI_ch2=0
    def ddbgfun_ch1(self):
        self.ddbg_ch1=0
    def ddbgfun_ch2(self):
        self.ddbg_ch2=0
        
    def ddxfun_ch1(self):
        sigma=self.sigma0ch1
        I=self.theta[2]
        self.ddx_ch1=(I*self.R/(np.sqrt(2)*sigma**3))*((self.xx-self.theta[0]-self.halfpixel)*np.exp(-(self.xx-self.theta[0]-self.halfpixel)**2/(2*self.sigma0ch1**2))-(self.xx-self.theta[0]+self.halfpixel)*np.exp(-(self.xx-self.theta[0]+self.halfpixel)**2/(2*self.sigma0ch1**2)))*self.Ey
    
    def ddxfun_ch2(self):
        sigma=self.sigma0ch1
        I=self.theta[2]
        self.ddx_ch2=(I*self.R2/(np.sqrt(2)*sigma**3))*((self.xx-self.theta[0]-self.halfpixel)*np.exp(-(self.xx-self.theta[0]-self.halfpixel)**2/(2*self.sigma0ch1**2))-(self.xx-self.theta[0]+self.halfpixel)*np.exp(-(self.xx-self.theta[0]+self.halfpixel)**2/(2*self.sigma0ch1**2)))*self.Ey
    
    def ddyfun_ch1(self):
        sigma=self.sigma0ch1
        I=self.theta[2]
        self.ddy_ch1=(I*self.R/(np.sqrt(2)*sigma**3))*((self.yy-self.theta[1]-self.halfpixel)*np.exp(-(self.yy-self.theta[1]-self.halfpixel)**2/(2*self.sigma0ch1**2))-(self.yy-self.theta[1]+self.halfpixel)*np.exp(-(self.yy-self.theta[1]+self.halfpixel)**2/(2*self.sigma0ch1**2)))*self.Ex
    
    def ddyfun_ch2(self):
        sigma=self.sigma0ch1
        I=self.theta[2]
        self.ddy_ch2=(I*self.R2/(np.sqrt(2)*sigma**3))*((self.yy-self.theta[1]-self.halfpixel)*np.exp(-(self.yy-self.theta[1]-self.halfpixel)**2/(2*self.sigma0ch1**2))-(self.yy-self.theta[1]+self.halfpixel)*np.exp(-(self.yy-self.theta[1]+self.halfpixel)**2/(2*self.sigma0ch1**2)))*self.Ex
    
    def CRLBfun(self,theta):
        self.theta=theta
        #self.sigmafun_ch1()
        #self.dsigmadzfun_ch1()
        self.Exfun()
        self.Eyfun()
        self.ukfun_ch1()
        self.dxfun_ch1()
        self.dyfun_ch1()
        self.dIfun_ch1()
        self.dbgfun_ch1()

        
        darray_ch1=np.zeros([4,self.roi,self.roi])
        darray_ch1[0]=self.dx_ch1
        darray_ch1[1]=self.dy_ch1

        darray_ch1[2]=self.dI_ch1
        darray_ch1[3]=self.dbg_ch1
        
        self.Exfun()
        self.Eyfun()
        self.ukfun_ch2()
        self.dxfun_ch2()
        self.dyfun_ch2()
        self.dIfun_ch2()
        self.dbgfun_ch2()

        darray_ch2=np.zeros([4,self.roi,self.roi])
        darray_ch2[0]=self.dx_ch2
        darray_ch2[1]=self.dy_ch2
        #darray_ch2[2]=self.dz_ch2
        darray_ch2[2]=self.dI_ch2
        darray_ch2[3]=self.dbg_ch2
        self.Fisher=np.zeros([4,4])
        for i in range(4):
            for j in range(4):
                self.Fisher[i,j]=np.sum((1/self.uk_ch1)*darray_ch1[i]*darray_ch1[j]+(1/self.uk_ch2)*darray_ch2[i]*darray_ch2[j])
        self.CRLB=np.sqrt(np.diag(np.linalg.inv(self.Fisher)))
    
    def theta_update(self,img_ch1,img_ch2):
        
        self.Exfun()
        self.Eyfun()
        self.ukfun_ch1()
        self.dxfun_ch1()
        self.dyfun_ch1()
        self.dIfun_ch1()
        self.dbgfun_ch1()
       
        
        self.ddxfun_ch1()
        self.ddyfun_ch1()
        self.ddIfun_ch1()
        self.ddbgfun_ch1()
      
        
        darray_ch1=np.zeros([4,self.roi,self.roi])
        darray_ch1[0]=self.dx_ch1
        darray_ch1[1]=self.dy_ch1
        darray_ch1[2]=self.dI_ch1
        darray_ch1[3]=self.dbg_ch1
        
        ddarray_ch1=np.zeros([4,self.roi,self.roi])
        ddarray_ch1[0]=self.ddx_ch1
        ddarray_ch1[1]=self.ddy_ch1
        ddarray_ch1[2]=self.ddI_ch1
        ddarray_ch1[3]=self.ddbg_ch1
        
        
        self.Exfun()
        self.Eyfun()
        self.ukfun_ch2()
        self.dxfun_ch2()
        self.dyfun_ch2()
        self.dIfun_ch2()
        self.dbgfun_ch2()
        #self.dzfun_ch2()
        
        self.ddxfun_ch2()
        self.ddyfun_ch2()
        self.ddIfun_ch2()
        self.ddbgfun_ch2()
        #self.ddzfun_ch2()
        
        darray_ch2=np.zeros([4,self.roi,self.roi])
        darray_ch2[0]=self.dx_ch2
        darray_ch2[1]=self.dy_ch2
        #darray_ch2[2]=self.dz_ch2
        darray_ch2[2]=self.dI_ch2
        darray_ch2[3]=self.dbg_ch2
        
        ddarray_ch2=np.zeros([4,self.roi,self.roi])
        ddarray_ch2[0]=self.ddx_ch2
        ddarray_ch2[1]=self.ddy_ch2
        #ddarray_ch2[2]=self.ddz_ch2
        ddarray_ch2[2]=self.ddI_ch2
        ddarray_ch2[3]=self.ddbg_ch2
        
        
        m1=(img_ch1/self.uk_ch1)-1.0
        m2=(img_ch2/self.uk_ch2)-1.0
        
        dm1=-(img_ch1/self.uk_ch1**2)
        dm2=-(img_ch2/self.uk_ch2**2)
        self.jacobian=np.zeros([4])
        for i in range(4):
            self.jacobian[i]=np.sum(darray_ch1[i]*m1+darray_ch2[i]*m2)
        #print(self.jacobian)
        self.Hessian=np.zeros([4,4])
        for i in range(4):
            for j in range(4):
                #self.Hessian[i,j]=np.sum(ddarray_ch1[i]*m1-darray_ch1[i]**2*((img_ch1/self.uk_ch1**2))+ddarray_ch2[i]*m2-darray_ch2[i]**2*((img_ch2/self.uk_ch2**2)))
                
                self.Hessian[i,j]=np.sum(darray_ch1[i]*darray_ch1[j]*dm1+darray_ch2[i]*darray_ch2[j]*dm2)


    def theta_update_one_ch_sigma_gpu(self,img_ch1):
        

        l=len(self.theta)
        
        
        self.ukfun_sigma_gpu()
        self.dxysigmafun_gpu()

        darray_ch1=torch.zeros([5,l,self.roi,self.roi],device=self.cuda0)
        
        darray_ch1[0]=self.dx_ch1
        darray_ch1[1]=self.dy_ch1
        darray_ch1[2]=self.dI_ch1
        darray_ch1[3]=self.dbg_ch1
        darray_ch1[4]=self.dsigma

        m1=(img_ch1/self.uk_ch1)-1.0
        
        dm1=-(img_ch1/self.uk_ch1**2)
        self.jacobian=torch.zeros([l,5],device=self.cuda0)
        for i in range(5):
            self.jacobian[:,i]=torch.sum(darray_ch1[i]*m1,(1,2))

        self.Hessian=torch.zeros([l,5,5],device=self.cuda0)
        self.fisher=torch.zeros([l,5,5],device=self.cuda0)
        for i in range(5):
            t=darray_ch1[i]*dm1
            for j in range(5):
                
                self.Hessian[:,i,j]=torch.sum(darray_ch1[j]*t,(1,2))
                self.fisher[:,i,j]+=torch.sum(darray_ch1[i]*darray_ch1[j]*(1.0/self.uk_ch1),(1,2))
    
    
    
    def theta_update_one_ch_gpu(self,img_ch1):
        

        l=len(self.theta)
        
        
        self.ukfun_ch1_gpu()
        self.dxyfun_ch1_gpu()
 
       
        darray_ch1=torch.zeros([4,l,self.roi,self.roi],device=self.cuda0)
        
        darray_ch1[0]=self.dx_ch1
        darray_ch1[1]=self.dy_ch1
        darray_ch1[2]=self.dI_ch1
        darray_ch1[3]=self.dbg_ch1

        m1=(img_ch1/self.uk_ch1)-1.0
        
        dm1=-(img_ch1/self.uk_ch1**2)
        self.jacobian=torch.zeros([l,4],device=self.cuda0)
        for i in range(4):
            self.jacobian[:,i]=torch.sum(darray_ch1[i]*m1,(1,2))

        self.Hessian=torch.zeros([l,4,4],device=self.cuda0)
        self.fisher=torch.zeros([l,4,4],device=self.cuda0)
        for i in range(4):
            t=darray_ch1[i]*dm1
            for j in range(4):
                
                self.Hessian[:,i,j]=torch.sum(darray_ch1[j]*t,(1,2))
                self.fisher[:,i,j]+=torch.sum(darray_ch1[i]*darray_ch1[j]*(1.0/self.uk_ch1),(1,2))
    
    
    def theta_update_one_ch(self,img_ch1):
        
        #self.Exfun()
        #self.Eyfun()
        self.ukfun_ch1()
        self.dxfun_ch1()
        self.dyfun_ch1()
        self.dIfun_ch1()
        self.dbgfun_ch1()
       
        darray_ch1=np.zeros([4,self.roi,self.roi])
        darray_ch1[0]=self.dx_ch1
        darray_ch1[1]=self.dy_ch1
        darray_ch1[2]=self.dI_ch1
        darray_ch1[3]=self.dbg_ch1
        

        m1=(img_ch1/self.uk_ch1)-1.0
        
        dm1=-(img_ch1/self.uk_ch1**2)
        self.jacobian=np.zeros([4])
        for i in range(4):
            self.jacobian[i]=np.sum(darray_ch1[i]*m1)
        #print(self.jacobian)
        if True:
            self.Hessian=np.zeros([4,4])
            for i in range(4):
                t=darray_ch1[i]*dm1
                for j in range(4):
                    #self.Hessian[i,j]=np.sum(ddarray_ch1[i]*m1-darray_ch1[i]**2*((img_ch1/self.uk_ch1**2))+ddarray_ch2[i]*m2-darray_ch2[i]**2*((img_ch2/self.uk_ch2**2)))
                    
                    self.Hessian[i,j]=np.sum(darray_ch1[j]*t)
    
    
    
    def chi_test(self,img_ch1,img_ch2): 
        self.ukfun_ch1()
        self.ukfun_ch2()
        uk1=self.uk_ch1
        uk2=self.uk_ch2
        return np.sum((uk1-img_ch1)**2/uk1+(uk2-img_ch2)**2/uk2)
    
    def chi_test_batch_second_ch(self,img_ch2,x_arr,y_arr,I_arr,bg_arr): 
        batch_size=len(img_ch2)
        chi=0.0
        for k in range(batch_size):
            self.theta=np.array([x_arr[k]+self.x_align,y_arr[k]+self.y_align,I_arr[k],bg_arr[k]])
            self.Exfun()
            self.Eyfun()
            self.ukfun_ch2()
            uk2=self.uk_ch2
            chi+=np.sum((uk2-img_ch2)**2/uk2)
        return chi
    
    def chi_test_batch_second_ch_scale_trans(self,img_ch2,x_arr,y_arr,I_arr,bg_arr,gx_arr,gy_arr): 
        batch_size=len(img_ch2)
        chi=0.0
        for k in range(batch_size):
            self.theta=np.array([(self.sx-1.0)*(x_arr[k]+gx_arr[k])+x_arr[k]+self.x_align,(self.sy-1.0)*(y_arr[k]+gy_arr[k])+y_arr[k]+self.y_align,I_arr[k],bg_arr[k]])

            #self.Exfun()
            #self.Eyfun()
            self.ukfun_ch2()
            uk2=self.uk_ch2
            chi+=np.sum((uk2-img_ch2)**2/uk2)
        return chi
    
    def chi_test_batch_second_ch_scale_trans_withr(self,img_ch2,x_arr,y_arr,I_arr,bg_arr,gx_arr,gy_arr,r,bg2): 
        batch_size=len(img_ch2)
        chi=0.0
        for k in range(batch_size):
            self.theta=np.array([(self.sx-1.0)*(x_arr[k]+gx_arr[k])+x_arr[k]+self.x_align,(self.sy-1.0)*(y_arr[k]+gy_arr[k])+y_arr[k]+self.y_align,I_arr[k],bg_arr[k],r[k],bg2[k]])
            self.R=r[k]
            self.R2=1.0-r[k]
            self.Exfun()
            self.Eyfun()
            self.ukfun_ch2_nolink()
            uk2=self.uk_ch2
            chi+=np.sum((uk2-img_ch2)**2/uk2)
        return chi
    
    def chi_test_one_ch_gpu(self,img_ch1): 
        self.ukfun_ch1_gpu()
        uk1=self.uk_ch1
        return torch.sum((uk1-img_ch1)**2/uk1,(1,2))
    
    def chi_test_one_ch_sigma_gpu(self,img_ch1): 
        self.ukfun_sigma_gpu()
        uk1=self.uk_ch1
        return torch.sum((uk1-img_ch1)**2/uk1,(1,2))
    
    def chi_test_two_ch_gpu(self,img_ch1,img_ch2): 
        self.ukfun_twoch_gpu()
        uk1=self.uk_ch1
        uk2=self.uk_ch2
        return torch.sum((uk1-img_ch1)**2/uk1,(1,2))+torch.sum((uk2-img_ch2)**2/uk2,(1,2))
    
    def chi_test_batch_second_chi_trans_gpu(self,img_ch2): 
        self.uk_fun_trans_align()
        uk2=self.uk_ch2
        return torch.sum((uk2-img_ch2)**2/uk2,(0,1,2))
    
    def chi_test_batch_second_chi_trans_scale_gpu(self,img_ch2): 
        self.uk_fun_scale_trans_align()
        uk2=self.uk_ch2
        return torch.sum((uk2-img_ch2)**2/uk2,(0,1,2))
    
    def chi_test_one_ch(self,img_ch1): 
        self.ukfun_ch1()
        uk1=self.uk_ch1
        return np.sum((uk1-img_ch1)**2/uk1)
    
    def LM_alg(self,init_theta,img_ch1,img_ch2):
        self.lam=0.1
        c=self.chi_test(img_ch1,img_ch2)
        self.theta=init_theta
        self.theta_update(img_ch1,img_ch2)
        
        dtheta=np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))@self.jacobian 
        
        self.theta[:4]=self.theta[:4]-dtheta
        cnew=self.chi_test(img_ch1,img_ch2)
        if (cnew<c):
            self.lam=self.lam*0.1
        elif (cnew>c):
            self.lam=self.lam*10
            self.theta[:4]=self.theta[:4]+dtheta
        matheta=dtheta
        for k in range(20):
            c=self.chi_test(img_ch1,img_ch2)
            
            self.theta_update(img_ch1,img_ch2)
            
            #dtheta=self.jacobian @ np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))
            dtheta=np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))@self.jacobian 

            dd=np.zeros([4])
            for i in range(4):
                if (dtheta[i]*matheta[i]>0):
                    dd[i]=dtheta[i]*(1.0/(1+abs(dtheta[i]/matheta[i])))
                    self.theta[i]=self.theta[i]-dd[i]
                else:
                    dd[i]=dtheta[i]*(1.0/(1+0.5*abs(dtheta[i]/matheta[i])))
                    self.theta[i]=self.theta[i]-dd[i]
                
            cnew=self.chi_test(img_ch1,img_ch2)
            if (cnew<c):
                self.lam=self.lam*0.1
                for i in range(4):
                    if dtheta[i]>matheta[i]:
                        matheta[i]=dtheta[i]
            elif (cnew>c and cnew<1.5*c):
                self.lam=self.lam
                for i in range(4):
                    if dtheta[i]>matheta[i]:
                        matheta[i]=dtheta[i]
            elif (cnew>c*1.5):
                self.lam=self.lam*10
                self.theta[:4]=self.theta[:4]+dd
    
                
   
    
   
    
    def LM_alg_one_ch_sigma_gpu(self,init_theta,img_ch1_cpu,peak_listarr):
        #print(init_theta[:,0])
        
        l=len(img_ch1_cpu)
        self.xxgpu = self.xxg.unsqueeze(0).repeat(l, 1, 1)
        self.yygpu = self.yyg.unsqueeze(0).repeat(l, 1, 1)
        self.R=1.0
        trouble=torch.ones([l],device=self.cuda0)
        #img_ch1=self.img_ch1
        img_ch1=torch.tensor(img_ch1_cpu,device=self.cuda0).to(torch.float32)
        
        #self.xxgpu = self.xxg.unsqueeze(0).repeat(l, 1, 1)
        #self.yygpu = self.yyg.unsqueeze(0).repeat(l, 1, 1)
        
        self.lam=0.001*torch.ones([l,5,5],device=self.cuda0)
        #self.theta=torch.tensor(init_theta,device=self.cuda0)
        self.theta=init_theta
        c=self.chi_test_one_ch_sigma_gpu(img_ch1)
        
        self.theta_update_one_ch_sigma_gpu(img_ch1)
        
        notfinsh=True
        while notfinsh:
            try:
                #self.hm=self.Hessian+self.lam*torch.diag_embed(torch.diagonal(self.Hessian, offset=0, dim1=1, dim2=2))
                self.hm=self.Hessian+self.lam*torch.diag(torch.ones([5],device=self.cuda0))
                     
                ind=(trouble ==0).nonzero()
                
                self.hm[ind]=torch.diag(torch.ones([5],device=self.cuda0))
                dtheta=torch.matmul(torch.linalg.inv(self.hm),self.jacobian.unsqueeze(2)) 
                notfinsh=False
            except: # work on python 3.x
                #print("wow")
                a=torch.linalg.matrix_rank(self.hm)
                ind=(a!=5).nonzero()
                
                trouble[ind]=0

        dtheta=dtheta[:,:,0]
        dtheta*=0.5
        
        
        self.theta[:,:5]=self.theta[:,:5]-dtheta
        cnew=self.chi_test_one_ch_sigma_gpu(img_ch1)
        l=torch.where(cnew<c,0.1,10).unsqueeze(1).unsqueeze(2).repeat(1, 5, 5)
        self.lam=self.lam*l
        a=torch.where(cnew<c,0,1).unsqueeze(1).repeat(1, 5)
        self.theta[:,:5]=self.theta[:,:5]+dtheta*a
        
        matheta=dtheta*1
        for k in range(30):
            
            c=self.chi_test_one_ch_sigma_gpu(img_ch1)
            
            self.theta_update_one_ch_sigma_gpu(img_ch1)
            
            #dtheta=self.jacobian @ np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))
            #dtheta=np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))@self.jacobian 
            notfinsh=True
            while notfinsh:
                try:
                    #self.hm=self.Hessian+self.lam*torch.diag_embed(torch.diagonal(self.Hessian, offset=0, dim1=1, dim2=2))
                    self.hm=self.Hessian+self.lam*torch.diag(torch.ones([5],device=self.cuda0))
                         
                    ind=(trouble ==0).nonzero()
                    
                    self.hm[ind]=torch.diag(torch.ones([5],device=self.cuda0))
                    #print(self.hm[ind[0]])
                    dtheta=torch.matmul(torch.linalg.inv(self.hm),self.jacobian.unsqueeze(2)) 
                    notfinsh=False
                except: # work on python 3.x
                    a=torch.linalg.matrix_rank(self.hm)
                    ind=(a!=5).nonzero()
                    #print(ind)
                    trouble[ind]=0

            dtheta*=0.5
            dtheta=dtheta[:,:,0]
            #print(dtheta.shape)
            dd=torch.where(dtheta*matheta>0,dtheta*(1.0/abs(dtheta/matheta)),dtheta*(1.0/abs(0.5*dtheta/matheta)))
            self.theta[:,:5]=self.theta[:,:5]-dd
            
            cnew=self.chi_test_one_ch_sigma_gpu(img_ch1)
            l=torch.where(cnew<c,0.1,1).unsqueeze(1).unsqueeze(2).repeat(1, 5, 5)
            l1=torch.where(cnew>c*1.5,1.0,10).unsqueeze(1).unsqueeze(2).repeat(1, 5, 5)
            self.lam=self.lam*l*l1
            
            matheta=torch.where(cnew.unsqueeze(1).repeat(1,5)<1.5*c.unsqueeze(1).repeat(1,5),dtheta,matheta)
            a=torch.where(cnew<c,0,1).unsqueeze(1).repeat(1, 5)
            self.theta[:,:5]=self.theta[:,:5]+dd*a
        
        a=torch.linalg.matrix_rank(self.fisher)
        ind=(a!=5).nonzero()
        trouble[ind]=0
        torch.cuda.synchronize()
        peak_listarr=peak_listarr.to(torch.float32)
        
        self.chi=c[trouble.cpu()==1].cpu()
        self.theta=self.theta.to("cpu")
        self.theta=self.theta[trouble.cpu()==1]
        self.theta[:,0]=(self.theta[:,0])+peak_listarr[trouble.cpu()==1,1]*self.pixelsize
        self.theta[:,1]=(self.theta[:,1])+peak_listarr[trouble.cpu()==1,0]*self.pixelsize
        self.frame=peak_listarr[trouble.cpu()==1,2]
        self.crlb=torch.sqrt(torch.diagonal(torch.linalg.inv(self.fisher[trouble.cpu()==1]),dim1=1,dim2=2)).cpu()
    
    
    def LM_alg_one_ch_gpu(self,init_theta,img_ch1_cpu,peak_listarr):
        #print(init_theta[:,0])
        
        l=len(img_ch1_cpu)
        
        trouble=torch.ones([l],device=self.cuda0)
        img_ch1=self.img_ch1
        
        
        #self.xxgpu = self.xxg.unsqueeze(0).repeat(l, 1, 1)
        #self.yygpu = self.yyg.unsqueeze(0).repeat(l, 1, 1)
        
        self.lam=0.0001*torch.ones([l,4,4],device=self.cuda0)
        #self.theta=torch.tensor(init_theta,device=self.cuda0)
        self.theta=init_theta
        c=self.chi_test_one_ch_gpu(img_ch1)
        
        self.theta_update_one_ch_gpu(img_ch1)
        
        notfinsh=True
        while notfinsh:
            try:
                #self.hm=self.Hessian+self.lam*torch.diag_embed(torch.diagonal(self.Hessian, offset=0, dim1=1, dim2=2))
                self.hm=self.Hessian+self.lam*torch.diag(torch.ones([4],device=self.cuda0))
                                
                ind=(trouble ==0).nonzero()
                
                self.hm[ind]=torch.diag(torch.ones([4],device=self.cuda0))
                dtheta=torch.matmul(torch.linalg.inv(self.hm),self.jacobian.unsqueeze(2)) 
                notfinsh=False
            except: # work on python 3.x
                #print("wow")
                a=torch.linalg.matrix_rank(self.hm)
                ind=(a!=4).nonzero()
                
                trouble[ind]=0
        
        #dtheta=torch.matmul(torch.linalg.inv(self.Hessian+self.lam*torch.diag_embed(torch.diagonal(self.Hessian, offset=0, dim1=1, dim2=2))),self.jacobian.unsqueeze(2)) 
        #dtheta=torch.matmul(torch.linalg.inv(self.Hessian+self.lam*torch.diag(torch.ones([4],device=self.cuda0))),self.jacobian.unsqueeze(2)) 
        
        dtheta=dtheta[:,:,0]
        dtheta*=0.5
        
        
        self.theta[:,:4]=self.theta[:,:4]-dtheta
        cnew=self.chi_test_one_ch_gpu(img_ch1)
        l=torch.where(cnew<c,0.1,10).unsqueeze(1).unsqueeze(2).repeat(1, 4, 4)
        self.lam=self.lam*l
        a=torch.where(cnew<c,0,1).unsqueeze(1).repeat(1, 4)
        self.theta[:,:4]=self.theta[:,:4]+dtheta*a
        
        matheta=dtheta*1
        for k in range(30):
            
            c=self.chi_test_one_ch_gpu(img_ch1)
            
            self.theta_update_one_ch_gpu(img_ch1)
            
            #dtheta=self.jacobian @ np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))
            #dtheta=np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))@self.jacobian 
            notfinsh=True
            while notfinsh:
                try:
                    #self.hm=self.Hessian+self.lam*torch.diag_embed(torch.diagonal(self.Hessian, offset=0, dim1=1, dim2=2))
                    self.hm=self.Hessian+self.lam*torch.diag(torch.ones([4],device=self.cuda0))
                     
                    ind=(trouble ==0).nonzero()
                    
                    self.hm[ind]=torch.diag(torch.ones([4],device=self.cuda0))
                    #print(self.hm[ind[0]])
                    dtheta=torch.matmul(torch.linalg.inv(self.hm),self.jacobian.unsqueeze(2)) 
                    notfinsh=False
                except: # work on python 3.x
                    a=torch.linalg.matrix_rank(self.hm)
                    ind=(a!=4).nonzero()
                    #print(ind)
                    trouble[ind]=0
                    #print(trouble)
                
            #dtheta=torch.matmul(torch.linalg.inv(self.Hessian+self.lam*torch.diag(torch.ones([4],device=self.cuda0))),self.jacobian.unsqueeze(2)) 

            dtheta*=0.5
            dtheta=dtheta[:,:,0]
            #print(dtheta.shape)
            dd=torch.where(dtheta*matheta>0,dtheta*(1.0/abs(dtheta/matheta)),dtheta*(1.0/abs(0.5*dtheta/matheta)))
            self.theta[:,:4]=self.theta[:,:4]-dd
            
            cnew=self.chi_test_one_ch_gpu(img_ch1)
            l=torch.where(cnew<c,0.1,1).unsqueeze(1).unsqueeze(2).repeat(1, 4, 4)
            l1=torch.where(cnew>c*1.5,1.0,10).unsqueeze(1).unsqueeze(2).repeat(1, 4, 4)
            self.lam=self.lam*l*l1
            
            matheta=torch.where(cnew.unsqueeze(1).repeat(1,4)<1.5*c.unsqueeze(1).repeat(1,4),dtheta,matheta)
            a=torch.where(cnew<c,0,1).unsqueeze(1).repeat(1, 4)
            self.theta[:,:4]=self.theta[:,:4]+dd*a
        
        
        a=torch.linalg.matrix_rank(self.fisher)
        ind=(a!=4).nonzero()
        trouble[ind]=0
        torch.cuda.synchronize()
        peak_listarr=peak_listarr.to(torch.float32)
        
        torch.cuda.synchronize()
        peak_listarr=peak_listarr.to(torch.float32)
        self.chi=c[trouble.cpu()==1].cpu()
        self.theta=self.theta.to("cpu")
        self.theta=self.theta[trouble.cpu()==1]
        self.theta[:,0]=(self.theta[:,0])+peak_listarr[trouble.cpu()==1,1]*self.pixelsize
        self.theta[:,1]=(self.theta[:,1])+peak_listarr[trouble.cpu()==1,0]*self.pixelsize
        self.frame=peak_listarr[trouble.cpu()==1,2]
        self.crlb=torch.sqrt(torch.diagonal(torch.linalg.inv(self.fisher[trouble.cpu()==1]),dim1=1,dim2=2)).cpu()
    
    

    def LM_alg_one_ch(self,init_theta,img_ch1):
        self.lam=0.1
        c=self.chi_test_one_ch(img_ch1)
        self.theta=init_theta
        self.theta_update_one_ch(img_ch1)
        
        dtheta=np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))@self.jacobian 
        dtheta*=0.5
        self.theta[:4]=self.theta[:4]-dtheta
        cnew=self.chi_test_one_ch(img_ch1)
        if (cnew<c):
            self.lam=self.lam*0.1
        elif (cnew>c):
            self.lam=self.lam*10
            self.theta[:4]=self.theta[:4]+dtheta
        matheta=dtheta
        for k in range(30):
            c=self.chi_test_one_ch(img_ch1)
            
            self.theta_update_one_ch(img_ch1)
            
            #dtheta=self.jacobian @ np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))
            dtheta=np.linalg.inv(self.Hessian+self.lam*np.diag(np.diag(self.Hessian)))@self.jacobian 
            dtheta*=0.5
            dd=np.zeros([4])
            for i in range(4):
                if (dtheta[i]*matheta[i]>0):
                    dd[i]=dtheta[i]*(1.0/(1+abs(dtheta[i]/matheta[i])))
                    self.theta[i]=self.theta[i]-dd[i]
                else:
                    dd[i]=dtheta[i]*(1.0/(1+0.5*abs(dtheta[i]/matheta[i])))
                    self.theta[i]=self.theta[i]-dd[i]
                
            cnew=self.chi_test_one_ch(img_ch1)
            if (cnew<c):
                self.lam=self.lam*0.1
                for i in range(4):
                    if dtheta[i]>matheta[i]:
                        matheta[i]=dtheta[i]
            elif (cnew>c and cnew<1.5*c):
                self.lam=self.lam
                for i in range(4):
                    if dtheta[i]>matheta[i]:
                        matheta[i]=dtheta[i]
            elif (cnew>c*1.5):
                self.lam=self.lam*10
                self.theta[:4]=self.theta[:4]+dd
        self.chi=c
        
    
    
                
    
    def Fisherfun_xyzIbg(self):
        #self.sigmafun_ch1()
        #self.dsigmadzfun_ch1()
        self.Exfun()
        self.Eyfun()
        self.ukfun_ch1()
        self.dxfun_ch1()
        self.dyfun_ch1()
        self.dIfun_ch1()
        self.dbgfun_ch1()
        self.dzfun_ch1()
        darray_ch1=np.zeros([5,self.roi,self.roi])
        darray_ch1[0]=self.dx_ch1
        darray_ch1[1]=self.dy_ch1
        darray_ch1[2]=self.dz_ch1
        darray_ch1[3]=self.dI_ch1
        darray_ch1[4]=self.dbg_ch1
        
        
        #self.sigmafun_ch2()
        #self.dsigmadzfun_ch2()
        self.Exfun()
        self.Eyfun()
        self.ukfun_ch2()
        self.dxfun_ch2()
        self.dyfun_ch2()
        self.dIfun_ch2()
        self.dbgfun_ch2()
        self.dzfun_ch2()
        darray_ch2=np.zeros([5,self.roi,self.roi])
        darray_ch2[0]=self.dx_ch2
        darray_ch2[1]=self.dy_ch2
        darray_ch2[2]=self.dz_ch2
        darray_ch2[3]=self.dI_ch2
        darray_ch2[4]=self.dbg_ch2
        
        
        self.Fisher=np.zeros([5,5])
        
        for i in range(5):
            for j in range(5):
                self.Fisher[i,j]=np.sum(darray_ch1[i]*darray_ch1[j]*(1.0/self.uk_ch1))+np.sum(darray_ch2[i]*darray_ch2[j]*(1.0/self.uk_ch2))
   
    def CRLB_xyzIbg(self):
        self.Fisherfun_xyzIbg()
        self.c=np.sqrt(np.diag(np.linalg.inv(self.Fisher)))
        return self.c
    
    def Fisherfun_xyIbg(self):
        
        self.Exfun()
        self.Eyfun()
        self.ukfun()
        self.dxfun()
        self.dyfun()
        self.dIfun()
        self.dbgfun()
        darray=np.zeros([4,self.roi,self.roi])
        darray[0]=self.dx
        darray[1]=self.dy
        darray[2]=self.dI
        darray[3]=self.dbg
        self.Fisher=np.zeros([4,4])
        
        for i in range(4):
            for j in range(4):
                self.Fisher[i,j]=np.sum(darray[i]*darray[j]*(1.0/self.uk))

    
    
    
    def init_theta_xyIbg(self,img_ch1_cpu):
        l=len(img_ch1_cpu)
        init_theta=torch.zeros([l,4],device=self.cuda0)
        self.img_ch1=torch.tensor(img_ch1_cpu,device=self.cuda0)
        self.xxgpu = self.xxg.unsqueeze(0).repeat(l, 1, 1)
        self.yygpu = self.yyg.unsqueeze(0).repeat(l, 1, 1)
        al=torch.sum(self.img_ch1,(1,2))
        init_theta[:,0]=torch.sum(self.xxgpu*self.img_ch1,(1,2))/al
        init_theta[:,1]=torch.sum(self.yygpu*self.img_ch1,(1,2))/al
        init_theta[:,3]=torch.median(torch.median(self.img_ch1,1)[0],1)[0]
        init_theta[:,2]=torch.sum(self.img_ch1-init_theta[:,3].unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi),(1,2))
        return init_theta
    
    def init_theta_xyIbgsigma(self,img_ch1_cpu,init_sigma):
        self.sqrtsigma=np.sqrt(2.0)*init_sigma
        self.sigma0ch1=init_sigma
        init_theta=self.init_theta_xyIbg(img_ch1_cpu)
        self.LM_alg_one_ch_gpu_subpixel(init_theta,img_ch1_cpu)
        self.img_ch1=self.img_ch1[self.ind[:,0]]
        l=len(self.ind[:,0])
        
        init_theta=torch.zeros([l,5],device=self.cuda0)
        init_theta[:,0]=self.theta[:,0]+self.roi//2*self.pixelsize
        init_theta[:,1]=self.theta[:,1]+self.roi//2*self.pixelsize
        init_theta[:,3]=torch.median(torch.median(self.img_ch1,1)[0],1)[0]
        init_theta[:,2]=torch.sum(self.img_ch1-init_theta[:,3].unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi),(1,2))
        init_theta[:,4]=init_sigma
        
        return init_theta,self.img_ch1
    
    def init_theta_xyIbgr(self,img_ch1_cpu,img_ch2_cpu):
        l=len(img_ch1_cpu)
        print(l)
        l=len(img_ch2_cpu)
        print(l)
        init_theta=torch.zeros([l,6],device=self.cuda0)
        self.img_ch1=torch.tensor(img_ch1_cpu,device=self.cuda0).to(torch.float32)
        self.img_ch2=torch.tensor(img_ch2_cpu,device=self.cuda0).to(torch.float32)
        self.xxgpu = self.xxg.unsqueeze(0).repeat(l, 1, 1)
        self.yygpu = self.yyg.unsqueeze(0).repeat(l, 1, 1)
        al=torch.sum(self.img_ch1,(1,2))
        init_theta[:,0]=torch.sum(self.xxgpu*self.img_ch1,(1,2))/al
        init_theta[:,1]=torch.sum(self.yygpu*self.img_ch1,(1,2))/al
        init_theta[:,3]=torch.median(torch.median(self.img_ch1,1)[0],1)[0]
        
        init_theta[:,4]=torch.median(torch.median(self.img_ch2,1)[0],1)[0]
        #init_theta[:,5]=torch.sum(self.img_ch1-init_theta[:,3].unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi),(1,2))
        I1=torch.sum(self.img_ch1-init_theta[:,3].unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi),(1,2))
        I2=torch.sum(self.img_ch2-init_theta[:,4].unsqueeze(1).unsqueeze(2).repeat(1, self.roi, self.roi),(1,2))
        init_theta[:,2]=I1+I2
        init_theta[:,5]=I1/(init_theta[:,2])
        self.img_ch1=img_ch1_cpu
        self.img_ch2=img_ch2_cpu
        if self.estimated_sigma:
            t,img_ch1_cpu1=self.init_theta_xyIbgsigma(img_ch1_cpu,120.0)
            
            img_ch2_cpu1=img_ch2_cpu[self.ind[:,0]]
            init_theta=init_theta[self.ind[:,0]]
            #self.R=1.0
            self.LM_alg_one_ch_sigma_gpu_subpixel(t,img_ch1_cpu1)
            img_ch1_cpu2=img_ch1_cpu1[self.trouble.cpu()==1]
            img_ch2_cpu2=img_ch2_cpu1[self.trouble.cpu()==1]
            init_theta=init_theta[self.trouble.cpu()==1]
            
            self.sigma=self.theta[:,4].clone().detach()
            self.img_ch1=img_ch1_cpu2
            self.img_ch2=img_ch2_cpu2
        return init_theta[self.ind[:,0]]
    
    def crop_spots(self,imgarr,peak_list,ROI):
        spot=[]
        ROIh=ROI//2
        for i in peak_list:
            spot_t=imgarr[i[2],i[0]-ROIh:i[0]+ROIh+1,i[1]-ROIh:i[1]+ROIh+1]
            
            spot.append(spot_t)
        return spot
    
    def crop_spots_ch2(self,imgarr,peak_list,ROI):
        spot=[]
        ROIh=ROI//2
        for i in peak_list:
            spot_t=imgarr[i[2],i[0]-ROIh-self.x_roughalign:i[0]+ROIh+1-self.x_roughalign,i[1]-ROIh-self.y_roughalign:i[1]+ROIh+1-self.y_roughalign]
            
            spot.append(spot_t)
        return spot
    
    def check_crop_spot(self,spot1,spot2,ROI,list_pos):
        for i in reversed(range(len(spot1))):
            x1,y1=spot1[i].shape
            x2,y2=spot2[i].shape
            if x1!= ROI or y1!=ROI or x2!= ROI or y2!=ROI:  
                spot1.pop(i)
                spot2.pop(i)
                list_pos.pop(i)
        return spot1,spot2,list_pos
                
    
    def select_channel2(self,imgarrch1,imgarrch2,peak_list):
        
        l=len(peak_list)
        ind=np.ones([l])
        for i in range(l):
            img=imgarrch2[i]
            bg=np.median(img)
            I_es=np.sum(img-bg)
            if I_es<100:
                ind[i]=0
        for i in reversed(range(l)):
            if ind[i]==0:
                imgarrch1.pop(i)
                imgarrch2.pop(i)
                peak_list.pop(i)
        return imgarrch1,imgarrch2,peak_list
                
    def imgfun_ch1(self):
        #r = (self.eng.dz_gaussian_withoutSAF(float(0),float(0),float(self.theta[2]),float(self.R*self.theta[2]),float(self.R*self.theta[3])))
        r = (self.eng.dz_gaussian_withoutSAF(float(0),float(0),float(self.theta[2]),float(self.R*self.theta[2]),float(self.R*self.theta[3])))
        self.img_ch1=(np.asarray(r['mu']))
        #self.dmudtheta_ch1=np.asarray(r['dmudtheta'])
    def imgfun_ch2(self):
        #r = (self.eng.dz_gaussian_withoutSAF(float(0),float(0),float(self.theta[2]+self.dz),float(self.R*self.theta[2]),float(self.R*self.theta[3])))
        r = (self.eng.dz_gaussian_withoutSAF(float(0),float(0),float(self.theta[2]+self.dz),float(self.R2*self.theta[2]),float(self.R2*self.theta[3])))
        self.img_ch2=(np.asarray(r['mu']))
        #self.dmudtheta_ch2=np.asarray(r['dmudtheta'])
    
    def CRLB_xyIbg(self):
        self.Fisherfun_xyIbg()
        self.c=np.sqrt(np.diag(np.linalg.inv(self.Fisher)))
        return self.c
    
    
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self=None




if __name__=="__main__":
    if True:
        #channel_roughalign = pickle.load( open( "channel_roughalign.p", "rb" ) )
        ROI=11
        #fn="C:/Users/localadmin/Downloads/OneDrive_1_8-1-2023/ExperimentalData/20210408_2ColorSTORM/aTub_A647_30ms_100LP_15x_1/aTub_A647_30ms_100LP_15x_1_MMStack_Default.tif"
        fn="C:/research/DMD-SIMFLUX/channel_transform/test1/cos7.tiff"
        imgarr=imread(fn).astype(np.float64)
        img_ch1=(imgarr[1:2000]).astype(np.float32)

        with GaussianPSF_biplane(roi=ROI) as gPSF:
            gPSF.pixelsize=108.0
            
            
            peak_list=spot_detection_method(roi=ROI).spot_detection_v3(img_ch1)
            
            spot=gPSF.crop_spots(img_ch1,peak_list,gPSF.roi)            
                
            gPSF.R=1.0
            gPSF.R2=1.0-gPSF.R
            result=[]
            
            batchsize=2000
            num=len(spot)//batchsize
            
            print("single molecule estimation")
            for i in tqdm(range(num)):

                subspot=spot[i*batchsize:(i+1)*batchsize]
                peak_listarr=torch.tensor(np.asarray(peak_list[i*batchsize:(i+1)*batchsize])).to(torch.float32)
                init_theta=gPSF.init_theta_xyIbg(subspot)
                gPSF.LM_alg_one_ch_gpu(init_theta,subspot,peak_listarr)
                for ii in range(len(gPSF.theta)):
                    result.append([gPSF.theta[ii,0],gPSF.theta[ii,1],gPSF.theta[ii,2],gPSF.theta[ii,3],120.0,gPSF.frame[ii],gPSF.chi[ii],gPSF.crlb[ii,0],gPSF.crlb[ii,1]])
            
            print("filtering")
            result=np.asarray(result)
            chi=result[:,6]
            chi=chi[chi>0]
            chi_thre=np.median(chi)*1.5
            result_filter=[]
            l=len(result)
            for i in range(l):
                add=True
                if  np.isnan(result[i,0]): #nan filter
                    #result_filter.append([result[i,0],result[i,1],result[i,2],result[i,3],result[i,4]])
                    add=False
                
                if result[i,6]>chi_thre: #chi square filter
                    add=False
                if result[i,2]<0 or result[i,3]<0: 
                    add=False
                if result[i,4]>250: #sigma filter
                    add=False
                if add:
                    result_filter.append([result[i,0],result[i,1],result[i,2],result[i,3],result[i,4],result[i,4],result[i,5],np.sqrt(result[i,7]**2+result[i,8]**2)])
    
            result_filter=np.asarray(result_filter)
            np.savetxt("localization_result_cos7_v2.csv", result_filter, delimiter=',', header="xnm,ynm,I,bg,sigmax,sigmax,frame,locprenm")
