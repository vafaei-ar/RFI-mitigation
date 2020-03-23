import matplotlib as mpl
mpl.use('agg')
import pylab as plt

import re
import sys
import glob 
#import h5py
import numpy as np
from PIL import Image
from astropy.io import fits
import h5py as h5
from random import shuffle

class BaseDataProvider(object):

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min #if a_min is not None else -np.inf
        self.a_max = a_max #if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, labels = self._next_data()
            
        data, labels = self._process(data, labels)
        nx = data.shape[1]
        ny = data.shape[0]

        return data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),
    
    def _process(self, data, label):

        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= self.a_min #np.amin(data)
        data /= (self.a_max-self.a_min) #np.amax(data)
    
        if self.n_class == 2:
#            nx = label.shape[1]
#            ny = label.shape[0]
#            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
#            labels[..., 1] = label
##            labels[..., 0] = ~label
#            labels[..., 0] = 1-label
            labels = np.stack([1-label,label],axis=-1)
            return data,labels
        
        return data,label
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y

def read_fits(filename):
    ext = filename.split('.')[-1]
    if ext=='fits':
        f = fits.open(filename)        
        fp = f[1].data
#        data,label = fp["DATA"].squeeze().T,fp['RFI_MASK'].squeeze().T
        data,label = fp["DATA"].squeeze().T,fp['RFI'].squeeze().T
        
        f.close()
    if ext=='h5':
        with h5py.File(filename, "r") as fp:
            data,label = np.array(fp['data']),np.array(fp['mask'])
    return data, label


def process(X,Y):
# (5, 100, 15, 4096, 4)
    X,Y = np.swapaxes(X,0,3),np.swapaxes(Y,0,3)
    X,Y = np.swapaxes(X,0,1),np.swapaxes(Y,0,1)
    X,Y = X.reshape(-1,4096,100,1),Y.reshape(-1,4096,100,1)
    return X,Y


class UnetDataProvider(BaseDataProvider):
    """
    Extends the BaseDataProvider to randomly select the next 
    data chunk
    """
    
    def __init__(self, nx, files, threshold, sim,
                 r_freq=500,
                 n_loaded=20,
                 a_min=30,
                 a_max=210,
                 phase_in=0,
                 channels=1,
                 n_class=1):
        super(UnetDataProvider, self).__init__(a_min, a_max)
        self.nx = nx
        self.files = files
        self.channels = channels
        self.n_class = n_class
        self.threshold = threshold
        self.sim = sim
        
        if phase_in and sim=='hide':
            assert 0,'HIDE simulations do not contain phase.'
        if phase_in==1 and channels!=2:
            assert 0,'channels should be 2!'
        self.phase_in = phase_in
        
        self.n_loaded = n_loaded
        self.r_freq = r_freq
        self.n_chunk = 0
        
        self.n_files = len(self.files)
        
        if self.n_loaded>self.n_files:
            print('Warning, number of files is less than requested loaded files.')
            print('Requested loaded files is changed to number of files.', self.n_files)
            self.n_loaded = self.n_files
        
        assert len(files) > 0, "No training files"
        
        self.reload()
#        print("Number of files used: %s"%len(files))
#        self._cylce_file()
    
#    def _read_chunk(self):
#        with h5py.File(self.files[self.file_idx], "r") as fp:
#            nx = fp["data"].shape[1]
#            idx = np.random.randint(0, nx - self.nx)
#            
#            sl = slice(idx, (idx+self.nx))
#            data = fp["data"][:, sl]
#            rfi = fp["mask"][:, sl]
#        return data, rfi

    def reload(self):
#        self.clean_list = [prefix+'clean_'+str(i)+'.h5' for i in range(100)]
#        self.dirty_list = [prefix+'dirty_'+str(i)+'.h5' for i in range(100)]  
        print('Reloading...')
        
        inds = np.arange(self.n_files)
        shuffle(inds)
        
        inds = inds[:self.n_loaded]
        print('Loaded files:',inds)

        if self.sim == 'hide':
            self.datas = []
            self.rfis = []
            
            for i in inds:
                data,rfi = read_fits(self.files[i])
                
                GET = True
                while GET:
                    if data.shape==(276, 14400) and rfi.shape==(276, 14400):
                        GET=False
                    else:
                        inds = np.arange(self.n_files)
                        shuffle(inds)
                        i = inds[0]
                        data,rfi = read_fits(self.files[i])
                
                self.datas.append(data)
                self.rfis.append(rfi)
                
            self.datas = np.array(self.datas)
            self.rfis = np.array(self.rfis)
            self.n_idx = self.datas.shape[0]

        
        elif self.sim == 'kat7':
        
            self.datas = []
            self.masks = []
            
            for i in inds:
                data = read_h5file(self.files[i],'data')
                mask = read_h5file(self.files[i],'mask')
            
                self.datas.append(data)
                self.masks.append(mask)  

            self.datas = np.array(self.datas)
            self.masks = np.array(self.masks)
            self.n_idx = self.datas.shape[0]

        elif self.sim == 'kat7fp':
        
            self.datas = []
            self.masks = []
            
            p = re.compile('\d\.h5')
            
            for i in inds:
                d = []
                m = []
                for j in range(4):
                    fname = p.sub(str(j)+'.h5', self.files[i])
                    d.append(read_h5file(fname,'data'))
                    m.append(read_h5file(fname,'mask'))
            
                data = np.stack(d,axis=-1)
#                mask = np.stack(m,axis=-1)
                mask = np.prod(m,axis=0)
                self.datas.append(data)
                self.masks.append(mask)

            self.datas = np.array(self.datas)
            self.masks = np.array(self.masks)
            self.n_idx = self.datas.shape[0]

        elif self.sim == 'mk':
        
            self.cleans = []
            self.dirties = []
            
            for i in inds:
                clean = read_h5file(self.files[i])
                dirty = read_h5file(self.files[i].replace('clean','dirty'))
            
                clean,dirty = process(clean,dirty)
            
                self.cleans.append(clean)
                self.dirties.append(dirty)  

            self.cleans = np.concatenate(self.cleans,axis=0)
            self.dirties = np.concatenate(self.dirties,axis=0)
            self.n_idx = self.cleans.shape[0]

#            if self.a_min is None:
#                self.a_min = np.min(self.cleans)
#            if self.a_max is None:
#                self.a_max = np.max(self.cleans)

        else:
            assert 0,'Unknown simumation in dp.'

    def _read_chunk(self):
        self.n_chunk += 1
        if self.r_freq:
            if self.n_chunk%self.r_freq==0:
                self.reload()
            
#        self.file_idx = np.random.choice(self.n_idx)
        inds = np.arange(self.n_idx)
        shuffle(inds)
        self.file_idx = inds[0]
#        print(self.n_idx,self.file_idx)
        
        if self.sim=='hide':
#            data,rfi = read_fits(self.files[self.file_idx])
#            data = self.datas[self.file_idx]
#            rfi = self.rfis[self.file_idx]
            
            data = self.datas[self.file_idx]
            rfi = self.rfis[self.file_idx]

            rfi = rfi>self.threshold      

#            rfi = np.abs(1.*rfi/data)>0.01     
            
            n0,nx = data.shape
            if n0%2!=0:
                n0 = n0-1
            if self.nx==0:
                return data,rfi
            idx = np.random.randint(0, nx - self.nx)  
            sl = slice(idx, (idx+self.nx))
            return data[:n0, sl],rfi[:n0, sl]
        
        elif self.sim=='kat7':
        
            data = self.datas[self.file_idx]
            mask = self.masks[self.file_idx]
        
            n0,nx = data.shape
            if n0%2!=0:
                n0 = n0-1
            if self.nx==0:
                return data,mask
            idx = np.random.randint(0, nx - self.nx)  
            sl = slice(idx, (idx+self.nx))
            return data[:n0, sl],mask[:n0, sl]        

        elif self.sim=='kat7fp':
        
            data = self.datas[self.file_idx]
            mask = self.masks[self.file_idx]
        
            n0,nx,nch = data.shape
            if n0%2!=0:
                n0 = n0-1
            if self.nx==0:
                return data,mask
            idx = np.random.randint(0, nx - self.nx)  
            sl = slice(idx, (idx+self.nx))
            return data[:n0, sl, :],mask[:n0, sl]   

        elif self.sim=='mk':
            
            clean = self.cleans[self.file_idx]
            dirty = self.dirties[self.file_idx]
            
            sigma = np.random.uniform(0.168,2*0.168)
            alpha = 2**np.random.uniform(-10,0)
            noise = complex_noise(clean,0,sigma)
            rfi = alpha * dirty
            dirty = clean + rfi + noise

            if self.phase_in==1:
                dirty = np.concatenate([np.absolute(dirty).astype(np.float32),
                                        np.angle(dirty).astype(np.float32)],axis=-1)
#                rfi = np.concatenate([np.absolute(rfi).astype(np.float32),
#                                      np.angle(rfi).astype(np.float32)],axis=-1)
            elif self.phase_in==2:
                dirty = np.angle(dirty).astype(np.float32)
#                rfi = np.angle(rfi).astype(np.float32)
            else:
                dirty = np.absolute(dirty).astype(np.float32)
#                rfi = np.absolute(rfi).astype(np.float32)
            
            rfi = np.absolute(rfi).astype(np.float32)                
#            else:
#                dirty = np.absolute(dirty)
#                rfi = np.absolute(rfi)
        
            rfi = rfi>self.threshold
        
            n0,nx,nch = dirty.shape
            if n0%2!=0:
                n0 = n0-1
            if self.nx==0:
                return dirty,rfi
            idx = np.random.randint(0, nx - self.nx)  
            sl = slice(idx, (idx+self.nx))
            return dirty[:n0, sl, :],rfi[:n0, sl, 0]        
        
        else:
            assert 0,'Unknown simumation in dp.'


    def _next_data(self):
        data, rfi = self._read_chunk()
#        nx = data.shape[1]
#        self._cylce_file()
        return data, rfi



    def check(self, n=1, prefix=''):
        x,y = self(n)
        for i in range(n):    
            if self.phase_in==1:
                fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,8))
                im1 = ax1.imshow(x[i,:,:,0],aspect='auto',norm=mpl.colors.LogNorm())
                ax1.set_title('x-abs')
                
                divider = make_axes_locatable(ax1)
                cax1 = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im1, cax=cax1, orientation='vertical')
                
                im2 = ax2.imshow(x[i,:,:,1],aspect='auto')
                ax2.set_title('x-ang')
                
                divider = make_axes_locatable(ax2)
                cax2 = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im2, cax=cax2, orientation='vertical')
                
                im3 = ax3.imshow(y[i,:,:,0],aspect='auto')
                ax3.set_title('y')
                
                divider = make_axes_locatable(ax3)
                cax3 = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im3, cax=cax3, orientation='vertical')                
        
            else:
                fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,5))
                im1 = ax1.imshow(x[i,:,:,0],aspect='auto',norm=mpl.colors.LogNorm())
                ax1.set_title('x')
                
                divider = make_axes_locatable(ax1)
                cax1 = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im1, cax=cax1, orientation='vertical')
                
                im2 = ax2.imshow(y[i,:,:,0],aspect='auto')
                ax2.set_title('y')
                
                divider = make_axes_locatable(ax2)
                cax2 = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im2, cax=cax2, orientation='vertical')
            
            plt.savefig(prefix+'dp_check'+str(i)+'.jpg')  
        
from mpl_toolkits.axes_grid1 import make_axes_locatable
#    def _cylce_file(self):
#        self.file_idx = np.random.choice(len(self.files))

def complex_noise(arr_in,mu,sigma,seed=0):
    #Function to create real and imaginary noise for an array.
    np.random.seed(seed)
    real_part = np.random.normal(mu,sigma,arr_in.shape)
    im_part = np.random.normal(mu,sigma,arr_in.shape)
    noise = real_part + 1j*im_part
    return noise

def open_h5file(filename):
    #function to read HDF5 file format
    return h5.File(filename,'r')

def read_h5file(filename,dataset='dataset'):
    #function to read a hdf5 file and output the array as a np.array
    data = open_h5file(filename)
    return data[dataset][()]




#dpt = UnetDataProvider(nx=nx,a_min=0, a_max=200, files=files_list)

'''
Created on Aug 18, 2016
original author: jakeret

Modified at: March 20, 2018
by:           yabebal fantaye

Modified on July 20, 2018
by: 		Anke van Dyk
'''

def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

class NgBaseDataProvider(object):
    
    def __init__(self,files,nx,ny,
                     a_min=-np.inf, a_max=np.inf,
                     rgb=False,verbose=0,channels=1,n_class=1):
        
        self.a_min = a_min 
        self.a_max = a_max 
        self.verbose = verbose
        self.nx = nx
        self.ny = ny
        self.rgb = rgb       
        self.files = files
        self.channels = channels
        self.n_class = n_class
        self.i_call = 0

        assert len(files) > 0, "No training files"
        if verbose: print("Number of files used: %s"%len(files))

    def read_chunk(self): 
        if self.i_call%25==0:
            filename = self.files[self.file_idx]
            data,self.label = np.load(filename)
            data = np.clip(np.fabs(data), self.a_min, self.a_max)
            data -= np.amin(data)
            data /= np.amax(data)
            self.data = data
            
        self.i_call += 1
        return self.data,self.label
           
    def _next_data(self):
        self.file_idx = np.random.choice(len(self.files))
        data, label = self.read_chunk()
        
        assert self.nx<=data.shape[0],'Error, Somthing is wrong is the given nx. Seems better to decrease it!'
        
        n_try = -1
        if self.ny>0:
            n_try += 1
            ny = data.shape[1]
            while ny < self.ny:
                print('Warning! something is wrong with {} dimensions.'.format(self.files[self.file_idx]))
                self.file_idx = np.random.choice(len(self.files))
                data, label = self.read_chunk()
                ny = data.shape[1]
                assert n_try<1000,'Error, Somthing is wrong is the given ny. Seems better to decrease it!'
            
        return data, label           

    def pre_process(self, data, label):
        data = np.expand_dims(data, axis=-1)
        if self.rgb:
            data = to_rgb(data).astype(np.uint8)
            
#        label -= np.amin(label)
#        label /= np.amax(label)

        if self.n_class == 1:
            return data,np.expand_dims(label, axis=-1)
        elif self.n_class == 2:
            lx,ly = label.shape
            labels = np.zeros((lx, ly, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = 1-label
            return data,labels
        else:
            assert 0,'Error, illegal number of class. It only can be either 1 or 2!'
    
    def __call__(self, n):
        
        data, label = self._next_data()
        data, label = self.pre_process(data, label)
        nx,ny,nc = data.shape   
        assert nc==self.channels,'Error, problem with given number of channel!'
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
        X[0] = data
        Y[0] = label
        for i in range(1,n):
            data, label = self._next_data()
            data, label = self.pre_process(data, label)
            X[i] = data
            Y[i] = label
    
        return X, Y
  
def get_slice(data,label,nx,ny):
    lx,ly = data.shape  
    if nx==0 or nx==lx:
        slx = slice(0, lx)                
    else:
        idx = np.random.randint(0, lx - nx)            
        slx = slice(idx, (idx+nx))       
    if ny==0 or ny==ly:
        sly = slice(0, ly)                
    else:
        idy = np.random.randint(0, ly - ny)            
        sly = slice(idy, (idy+ny))
    return data[slx, sly],label[slx, sly]
   
def threshold_mask(data,rfi,thresholds,th_labels=None):
    if not (isinstance(thresholds, list) or isinstance(thresholds, np.ndarray)):
        thresholds = [thresholds]

    n_trsh = len(thresholds)
    if th_labels is None:
        th_labels = np.arange(n_trsh)+1

#    rel_rfi = np.abs(1.*rfi/data)
    rel_rfi = np.abs(rfi)
    mask = np.zeros(data.shape)
    for i in range(n_trsh-1):
        mask[(thresholds[i]<rel_rfi) & (thresholds[i+1]>=rel_rfi)] = th_labels[i]        
    mask[thresholds[n_trsh-1]<rel_rfi] = th_labels[n_trsh-1]
    
#    if verbose:
#        txt='percentage of rfi pixels with >{:.3e} % RFI fraction: {:.2e} %'
#        print( txt.format(threshold,
#               np.divide(100.*np.sum(label), np.product(label.shape)) ))
    return mask

def read_chunk_hdf5(filename,nx,ny,label_tag,thresholds=None,th_labels=None,verbose=0):
#    if thresholds is not None:
#        thresholds = thresholds
#    else:
#        thresholds = [0.01]
        
    with h5py.File(filename, "r") as fp:
        column_names = list(fp.keys())
        if label_tag is None or not (label_tag in column_names):
            print('You did not select any label tag for mask production, please choose!',list(column_names))
            assert label_tag in column_names, 'Error, Tag is not found!'
        
        data,label = np.array(fp['data']),np.array(fp[label_tag])
        data,label = get_slice(data,label,nx,ny)
        if label_tag=='rfi_map' and thresholds is not None:
            label = threshold_mask(data,label,thresholds,th_labels=th_labels)
            
    return data, label

def read_chunk_sdfits(filename,nx,ny,label_tag,thresholds=None,th_labels=None,verbose=0):
#    if thresholds is not None:
#        thresholds = thresholds
#    else:
#        thresholds = [0.01]
    
    f = fits.open(filename)        
    fp = f[1].data
    column_names = fp.columns.names
    if label_tag is None or not (label_tag in column_names):
        print('You did not select any label tag for mask production, please choose!',column_names)
        assert 'Error, Tag is not found!'

    data,label = fp["DATA"].squeeze().T,fp[label_tag].squeeze().T
    data,label = get_slice(data,label,nx,ny)
    if label_tag=='RFI' and thresholds is not None:
        label = threshold_mask(data,label,thresholds,th_labels=th_labels)
    f.close()
    return data, label

class DataProvider(NgBaseDataProvider):
    
    def __init__(self,files,nx=0,ny=0, 
                     label_name=None,thresholds=None,th_labels=None,
                     a_min=-np.inf, a_max=np.inf,
                     rgb=False,verbose=0,channels=1,n_class=1):
        
        super(DataProvider, self).__init__(files, nx, ny,
                     a_min=a_min, a_max=a_max,
                     rgb=rgb,verbose=verbose,channels=channels,n_class=n_class)
    
        self.label_tag = label_name
        self.thresholds = thresholds
        self.th_labels = th_labels

    def read_chunk(self):     
        filename = self.files[self.file_idx]
        ext = filename.split('.')[-1]
        
        nx,ny = self.nx,self.ny
        label_tag = self.label_tag
        thresholds = self.thresholds
        th_labels = self.th_labels
        verbose = self.verbose
        
        if ext == 'fits':
            data,label = read_chunk_sdfits(filename,nx,ny,label_tag,thresholds=thresholds,th_labels=th_labels,verbose=verbose)
        elif ext == 'h5':
            data,label = read_chunk_hdf5(filename,nx,ny,label_tag,thresholds=thresholds,th_labels=th_labels,verbose=verbose)
        else:
            assert 0,'Error, unsupported file format!'
            
        return data,label       
        
        
        
        
#def read_chunk_hdf5(filename,nx,ny,label_tag,threshold=None,verbose=0):
#    if threshold is not None:
#        threshold = threshold
#    else:
#        threshold = 0.01
#        
#    with h5py.File(filename, "r") as fp:
#        column_names = fp.keys()
#        if label_tag is None or not (label_tag in column_names):
#            print('You did not select any label tag for mask production, please choose!',column_names)
#            assert 'Error, Tag is not found!'

#        lx,ly = fp["data"].shape

#        if nx==0 or nx==lx:
#            slx = slice(0, lx)                
#        else:
#            idx = np.random.randint(0, lx - nx)            
#            slx = slice(idx, (idx+nx))
#            
#        if ny==0 or ny==ly:
#            sly = slice(0, ly)                
#        else:
#            idy = np.random.randint(0, ly - ny)            
#            sly = slice(idy, (idy+ny))

#        data = fp["data"][slx, sly]
#        label = fp[label_tag][slx, sly]

#        if label_tag=='rfi_map':
#            label = 100*np.abs(label)>threshold
#            if verbose:
#                txt='percentage of rfi pixels with >{:.3f} % RFI fraction: {:.2f} %'
#                print( txt.format(threshold,
#                       np.divide(100.*np.sum(label), np.product(label.shape)) ))                
#    return data, label        
