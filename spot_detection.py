# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:59:48 2023

@author: localadmin
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

class spot_detection_method:
    def __init__(self,roi=21):
        self.roi=roi
    def spot_detection_v3(self,imgarr,threshold_rel=0.05,threshold_abs=1.5):
        
        from skimage.feature import peak_local_max
        from scipy.ndimage import gaussian_filter
    
        l,lx,ly=imgarr.shape
        peak_listall=[]
        print("spot detection for"+str(len(imgarr))+" frames")
        
        for kk in tqdm(range(len(imgarr)),position=0, leave=True):
            if kk%100==0:
                print(kk)
            im=imgarr[kk]
            data=gaussian_filter(im, sigma=1.5)-gaussian_filter(im, sigma=2.5)
            peak_list = peak_local_max(data, min_distance=7,threshold_rel=threshold_rel,threshold_abs=np.median(data)*threshold_abs)
            
            for i in range(len(peak_list)):
                peak_listall.append([peak_list[i, 0],peak_list[i, 1],kk])
        return peak_listall
    
    
        
        def find_dist(peak,th,lx,ly,img1):
            l=len(peak)
            c=np.ones([l])
            d=np.zeros([l,l])
            for i in range(l):
                for j in range(l):
                    d[i,j]=np.sqrt((peak[i][0]-peak[j][0])**2+(peak[i][1]-peak[j][1])**2)
            for i in range(l):
                for j in range(l):
                    if d[i,j]<th and d[i,j]>0.01:
                        c[i]=0
                        c[j]=0
            for i in range(l):
                if peak[i][0]<30 or peak[i][0]>lx-30 or peak[i][1]<30 or peak[i][1]>ly-30:
                    c[i]=0

            
            for i in reversed(range(l)):
                if c[i]==0:
                    peak.pop(i)

            return peak
        
        l,lx,ly=imgarr.shape
        peak_listall=[]
        for kk in range(len(imgarr)):
            peak_list=[]
            img=imgarr[kk]
            
            
            peak_list = detect_peaks(img)
           
            peak_list=find_dist(peak_list,self.roi//2+2,lx,ly,img)
            for i in peak_list:
                peak_listall.append([i[0],i[1],kk])
        return peak_listall
    
    
    def spot_detection(self,imgarr):
        from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
        from scipy.ndimage.filters import maximum_filter
        def detect_peaks(image):
            """
            Takes an image and detect the peaks usingthe local maximum filter.
            Returns a boolean mask of the peaks (i.e. 1 when
            the pixel's value is the neighborhood maximum, 0 otherwise)
            """
        
            # define an 8-connected neighborhood
            neighborhood = generate_binary_structure(2,2)
        
            #apply the local maximum filter; all pixel of maximal value 
            #in their neighborhood are set to 1
            #local_max = maximum_filter(image, footprint=neighborhood)==image
            local_max = maximum_filter(image, size=(3,3))==image
            #local_max is a mask that contains the peaks we are 
            #looking for, but also the background.
            #In order to isolate the peaks we must remove the background from the mask.
        
            #we create the mask of the background
            background = (image==0)
        
            #a little technicality: we must erode the background in order to 
            #successfully subtract it form local_max, otherwise a line will 
            #appear along the background border (artifact of the local maximum filter)
            eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        
            #we obtain the final mask, containing only peaks, 
            #by removing the background from the local_max mask (xor operation)
            detected_peaks = local_max ^ eroded_background
        
            return detected_peaks
        
        def find_dist(peak,th,lx,ly,img1):
            l=len(peak)
            c=np.ones([l])
            d=np.zeros([l,l])
            for i in range(l):
                for j in range(l):
                    d[i,j]=np.sqrt((peak[i][0]-peak[j][0])**2+(peak[i][1]-peak[j][1])**2)
            for i in range(l):
                for j in range(l):
                    if d[i,j]<th and d[i,j]>0.01:
                        c[i]=0
                        c[j]=0
            for i in range(l):
                if peak[i][0]<20 or peak[i][0]>lx-20 or peak[i][1]<20 or peak[i][1]>ly-20:
                    c[i]=0
    
            
            for i in reversed(range(l)):
                if c[i]==0:
                    peak.pop(i)
    
            return peak
        
        l,lx,ly=imgarr.shape
        peak_listall=[]
        for kk in range(len(imgarr)):
            if kk%10==0:
                print(kk)
            peak_list=[]
            img=imgarr[kk]
            img1=gaussian_filter(img, sigma=1.5)
            img1[img1<np.median(img1)]=0
            
            detected_peaks = detect_peaks(img1)
            
            for i in range(lx):
                for j in range(ly):
                    if detected_peaks[i,j]:
                        peak_list.append([i,j,kk])
            #print(len(peak_list))
            peak_list=find_dist(peak_list,self.roi//2+2,lx,ly,img)
            for i in peak_list:
                peak_listall.append(i)
        return peak_listall
    
    def spot_detection_v1(self,imgarr):
        from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
        from scipy.ndimage.filters import maximum_filter
        def detect_peaks(image,oriimg,kk):
            peak=[]
            
            lx,ly=image.shape
            find=True
            first=True
            t=image*1
            while first:
                i,j=np.unravel_index(t.argmax(), t.shape)
                if i>self.roi and i<lx-self.roi and j>self.roi and j<ly-self.roi:
    
    
                    sub=t[i-self.roi//2:i+self.roi//2+1,j-self.roi//2:j+self.roi//2+1]
                    maI=np.sum(sub)
                    
                    first=False
            find=True
            while find:
                i,j=np.unravel_index(t.argmax(), t.shape)
                if i>self.roi and i<lx-self.roi and j>self.roi and j<ly-self.roi:
                    sub=t[i-self.roi//2:i+self.roi//2+1,j-self.roi//2:j+self.roi//2+1]
                    I=np.sum(sub)
                    print(I)
                    if I>maI*0.25:
                        find=True
                        t[i-self.roi//2:i+self.roi//2+1,j-self.roi//2:j+self.roi//2+1]=0
                        peak.append([i,j,kk])
                    else:
                        find=False
    
            return peak
        
        def find_dist(peak,th,lx,ly,img1):
            l=len(peak)
            c=np.ones([l])
            d=np.zeros([l,l])
            for i in range(l):
                for j in range(l):
                    d[i,j]=np.sqrt((peak[i][0]-peak[j][0])**2+(peak[i][1]-peak[j][1])**2)
            for i in range(l):
                for j in range(l):
                    if d[i,j]<th and d[i,j]>0.01:
                        c[i]=0
                        c[j]=0
            for i in range(l):
                if peak[i][0]<30 or peak[i][0]>lx-30 or peak[i][1]<30 or peak[i][1]>ly-30:
                    c[i]=0
    
            
            for i in reversed(range(l)):
                if c[i]==0:
                    peak.pop(i)
          
            return peak
        
        l,lx,ly=imgarr.shape
        peak_listall=[]
        for kk in range(len(imgarr)):
            print(kk)
            peak_list=[]
            img=imgarr[kk]
            img1=gaussian_filter(img, sigma=1.0)-gaussian_filter(img, sigma=1.5)
            #img1[img1<np.median(img1)+np.std(img1)*3.0]=0
            
            peak_list = detect_peaks(img1,img,kk)
            #peak_list=find_dist(peak_list,self.roi//2,lx,ly,img)
            for i in peak_list:
                peak_listall.append(i)
        return peak_listall
    
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self=None
        