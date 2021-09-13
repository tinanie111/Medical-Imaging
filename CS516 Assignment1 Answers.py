# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:37:54 2021

@author: tina
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import numpy.fft as f

""" import and decompression the nii files """
img_car_axial = nib.load("E:/medical imaging/assignment 1/modalities/cardiac_axial.nii.gz")
img_car_realt = nib.load("E:/medical imaging/assignment 1/modalities/cardiac_realtime.nii.gz")
img_ct = nib.load("E:/medical imaging/assignment 1/modalities/ct.nii.gz")
img_fmti = nib.load("E:/medical imaging/assignment 1/modalities/fmri.nii.gz")
img_meanp = nib.load("E:/medical imaging/assignment 1/modalities/meanpet.nii.gz")
img_swi = nib.load("E:/medical imaging/assignment 1/modalities/swi.nii.gz")
img_T1_t = nib.load("E:/medical imaging/assignment 1/modalities/T1_with_tumor.nii.gz")
img_tof = nib.load("E:/medical imaging/assignment 1/modalities/tof.nii.gz")

""" Get the data """
img_car_axial = img_car_axial.get_data()
img_car_realt = img_car_realt.get_data()
img_ct = img_ct.get_data()
img_fmti = img_fmti.get_data()
img_meanp = img_meanp.get_data()
img_swi = img_swi.get_data()
img_T1_t = img_T1_t.get_data()
img_tof = img_tof.get_data()

""" Slice the 4-D images by the first volumn """
img_car_axial_vol0 = img_car_axial[:,:,:,0]
img_car_realt_vol0 = img_car_realt[:,:,:,0]
img_fmti_vol0 = img_fmti[:,:,:,0]

""" Define the lists of images and titles for furthur use"""
#3D image list
imglist = [img_car_axial_vol0,img_car_realt_vol0,img_ct,img_fmti_vol0,
           img_meanp,img_swi,img_T1_t,img_tof]
#original image list
imglist_t = [img_car_axial,img_car_realt,img_ct,img_fmti,
           img_meanp,img_swi,img_T1_t,img_tof]
imgtitle = ["cardiac_axial","cardiac_realtime","ct","fmri","meanpet","swi",
            "T1_with_tumor","tof"]


""" Part 1: Plot these images """
plt.figure()
plt.figure(dpi=500)
for i in range(8):
    plt.subplot(3,3,i+1)
    plt.axis('off')
    plt.title(imgtitle[i],fontsize=9)
    plt.imshow(imglist[i][:,:,int(imglist[i].shape[2]/2)], cmap='jet')
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.suptitle("Jet Images") 

""" Part 1: MIP and MIP for TOP """
#MIP for the swi
plt.figure()
plt.figure(dpi=500)
img_swi_min = np.min(img_swi[:,:,200:300],axis=2)
plt.subplot(1,2,1)
plt.axis('off')
plt.title('SWI MIP')
plt.imshow(img_swi_min)

#MIP for the TOP
img_swi_max = np.max(img_swi[:,:,200:300],axis=2)                        
plt.subplot(1,2,2)
plt.axis('off')
plt.title('TOF MIP')
plt.imshow(img_swi_max,cmap='jet')
plt.suptitle("MIPs") 

""" Part 2: Contrast estimation """
#Michelson
def findMichelson(imglist):
    michelsonlist = []
    for i in range(8):
        if np.min(imglist[i])>0:
            imgmin = np.min(imglist[i])
        else:
            imgmin = 0
        imgmax = np.max(imglist[i])
        Cmichelson = round((imgmax-imgmin)/(imgmax+imgmin),1)
        michelsonlist.append(Cmichelson)
    return michelsonlist
Cmichelson = findMichelson(imglist_t)

#RMS
def findRMS(imglist):
    RMSlist = []
    for i in range(8):
        RMS = round(np.sqrt(np.mean((imglist[i]-np.mean(imglist[i]))**2)),1)
        RMSlist.append(RMS)
    return RMSlist
Crms= findRMS(imglist_t)

#Entropy
def findEntropy(imglist):
    entropylist = []
    for i in range(8):
        imghist = np.histogram(imglist[i])
        imgfrq = imghist[0][np.nonzero(imghist[0])]
        imgpro = [float(h)/sum(imgfrq) for h in imgfrq]
        entropy = round(np.abs(np.sum(np.multiply(imgpro, np.log2(imgpro)))),1)
        entropylist.append(entropy)
    return entropylist
Centropy = findEntropy(imglist_t)

#Plot motalities with contrast
plt.figure()
plt.figure(dpi=500)
for i in range(8):
    plt.subplot(3,3,i+1)
    plt.axis('off')
    plt.title("{} \n Cmichelson:{} Crms:{} Centropy:{}".format\
              (imgtitle[i],Cmichelson[i],Crms[i],Centropy[i]),fontsize=4)
    plt.imshow(imglist[i][:,:,int(imglist[i].shape[2]/2)], cmap='jet')
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.suptitle("Modalities with Contrasts")

"""Part 3: SNRestimation, quantifying noise """

#SNR: We have to find and test some none zero noise patches, so in this 
#question, we didn't apply a "for" function to get all resaults.
img_car_axial_snr = np.mean(img_car_axial)\
    /np.std(img_car_axial_vol0[260:280,260:280,2])
img_car_realt_snr = np.mean(img_car_realt)\
    /np.std(img_car_realt_vol0[0:20,130:150,0])
img_ct_scr = np.mean(img_ct)/np.std(img_ct[60:80,60:80,119])
img_fmti_scr = np.mean(img_fmti)/np.std(img_fmti_vol0[0:20,0:20,18])
img_meanp_scr = np.mean(img_meanp)/np.std(img_meanp[40:60,40:60,103])
img_swi_scr = np.mean(img_swi)/np.std(img_swi[80:100,80:100,249])
img_T1_t_scr = np.mean(img_T1_t)/np.std(img_T1_t[0:20,0:20,127])
img_tof_scr = np.mean(img_tof)/np.std(img_tof[0:20,0:20,124])

SNRlist = [img_car_axial_snr,img_car_realt_snr,img_ct_scr,\
      img_fmti_scr,img_meanp_scr,img_swi_scr,img_T1_t_scr,img_tof_scr]
SNRlist = [np.abs(round(i,1)) for i in SNRlist]

for i in range(8):
    print("{}:{}".format(imgtitle[i],SNRlist[i]))
#As shown on the printed result, the CT has the lowest contrast while the 
#cardiac_axial is the highest.

#Plot the noise histogram with SNRs
positionlist = [img_car_axial_vol0[260:280,260:280,2],
                img_car_realt_vol0[0:20,130:150,0],
                img_ct[60:80,60:80,119],
                img_fmti_vol0[0:20,0:20,18],
                img_meanp[40:60,40:60,103],
                img_swi[80:100,80:100,249],
                img_T1_t[0:20,0:20,127],
                img_tof[0:20,0:20,124]]
plt.figure()
plt.figure(dpi=500)
for i in range(8):
    plt.subplot(3,3,i+1)
    plt.title("{}\nSNR:{}".format(imgtitle[i],SNRlist[i]),fontsize=6)
    plt.xticks([])
    plt.yticks(fontsize=5)
    plt.hist(positionlist[i])
plt.subplots_adjust(wspace=0.2, hspace=0.45)
plt.suptitle("Noise Histogram")

""" Part 4: Linear filtering """

def getgaussfilt(img,sigma):
    img_freqs = f.fftshift(f.fftn(img))
    [X,Y,Z] = np.mgrid[0:img.shape[0],0:img.shape[1],0:img.shape[2]]
    xpr,ypr,zpr = X-(img.shape[0])//2,Y-(img.shape[1])//2,Z-(img.shape[2])//2
    gaussfilt = np.exp(-((xpr**2+ypr**2+zpr**2)/(2*sigma**2)))/(2*np.pi*sigma**2)
    gaussfilt = gaussfilt/np.max(gaussfilt)
    filtered_freqs = img_freqs*gaussfilt
    filtered = np.abs(f.ifftn(f.fftshift(filtered_freqs)))
    return filtered

def plotgaussfilter(imglist,imgtitle,sigma):
    plt.figure(dpi=500)
    for i in range(8):
        plt.subplot(3,3,i+1)
        plt.axis('off')
        plt.title(imgtitle[i],fontsize=9)
        plt.imshow(getgaussfilt(imglist[i], sigma)[:,:,int(imglist[i].shape[2]/2-1)])
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.suptitle("GaussFilter, Sigma:{number}".format(number = sigma))

plt.figure()
plotgaussfilter(imglist,imgtitle,sigma=2)

plt.figure()
plotgaussfilter(imglist,imgtitle,sigma=4)

plt.figure()
plotgaussfilter(imglist,imgtitle,sigma=6)

