import os
import h5py as h5
import resource
import numpy as np
from time import time
from glob import glob
from sys import getsizeof
from random import shuffle
#from sklearn.utils import shuffle

#np.random.seed(0)

def open_h5file(filename):
    #function to read HDF5 file format
    return h5.File(filename,'r')

def read_h5file(filename,dataset='dataset'):
    #function to read a hdf5 file and output the array as a np.array
    data = open_h5file(filename)
    return data[dataset][()]

def save_h5file(savename,input_array,dataset='dataset'):
    #function to save hdf5 
    out = h5.File(savename, 'w')
    out.create_dataset(dataset, data=input_array)
    out.close()

def extract_complex_vis(filename):
    #function to extract complex dirty and clean visibility from chris sim data
    data = open_h5file(filename)
    group = data['output']
    group_clean = group['vis_clean']
    group_dirty = group['vis_dirty']
    return group_clean[()],group_dirty[()]

def smear(data,axis=0,check=0):
    n_time = data.shape[axis]
    nt = int(n_time//8)
    sdata = np.array(np.split(data, nt, axis=axis))
    #print(sdata.shape)

    ## TO CHECK THE SMEARING WORKS RIGHT.
    if check:
        for i in range(nt):
            dd = np.all(sdata[i,:,0,0,0]==data[i*8:i*8+8,0,0,0])
            if not dd:
                print(i)
                
    return np.mean(sdata,axis=axis+1)
    
def complex_noise(arr_in,mu,sigma,seed=0):
    #Function to create real and imaginary noise for an array.
    np.random.seed(seed)
    real_part = np.random.normal(mu,sigma,arr_in.shape)
    im_part = np.random.normal(mu,sigma,arr_in.shape)
    noise = real_part + 1j*im_part
    return noise

#x = read_h5file('shuf_firstdat_dirty_vis.h5',dataset='shuf_firstdat_dirty_vis')
#y = read_h5file('shuf_firstdat_rfi_vis.h5',dataset='shuf_firstdat_rfi_vis')

#def get_slice(x,y,nx,ny):
#    lx,ly = x.shape[:2]
#    if nx==0 or nx==lx:
#        slx = slice(0, lx)                
#    else:
#        idx = np.random.randint(0, lx - nx)            
#        slx = slice(idx, (idx+nx))       
#    if ny==0 or ny==ly:
#        sly = slice(0, ly)                
#    else:
#        idy = np.random.randint(0, ly - ny)            
#        sly = slice(idy, (idy+ny))
#    return x[slx, sly],y[slx, sly]

eps = 1e-4
def tfpnr(truth, pred):
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    npred = np.logical_not(pred)
    pos = 1.*np.sum(pred)
    neg = 1.*np.sum(npred)

    tp = np.sum(truth[pred]==True)
    #     fp = np.sum(truth[pred_mask]==False)
    fp = pos-tp
    tn = np.sum(truth[npred]==False)
    #     fn = np.sum(truth[~pred_mask]==True)
    fn = neg-tn

    return tp,fp,tn,fn
    


class Data_Provider(object):

    def __init__(self,nx,ny,n0,n1,a_min=0, a_max=200):

        prefix = '../../data/new_data/shuf_firstdat_'
        self.X = read_h5file(prefix+'dirty_vis.h5',dataset='shuf_firstdat_dirty_vis')
        self.Y = read_h5file(prefix+'rfi_vis.h5',dataset='shuf_firstdat_rfi_vis')
#        self.X = np.clip(np.fabs(self.X), a_min, a_max)
#        self.Y = np.clip(np.fabs(self.Y), a_min, a_max)
        self.nx = nx
        self.ny = ny
        self.n0 = n0
        self.n1 = n1
    
    def __call__(self,n):
        xp = []
        yp = []
        for i in range(n):
            nimg = np.random.randint(self.n0,self.n1)
            xx,yy = get_slice(self.X[nimg],self.Y[nimg],self.nx,self.ny)
            xp.append(xx)
            yp.append(yy)
     
        xp, yp = np.array(xp), np.array(yp)
        xp, yp = np.expand_dims(xp,-1), np.expand_dims(yp,-1)
        return xp, yp
    
    def getter(self,nimg):
        xp, yp = self.X[nimg],self.Y[nimg]
        xp, yp = np.array(xp), np.array(yp)
        xp, yp = np.expand_dims(xp,-1), np.expand_dims(yp,-1)
        xp, yp = np.expand_dims(xp,0), np.expand_dims(yp,0)
        return xp, yp



class Data_Provider_Jul_2019(object):

    def __init__(self,nx,ny,n0,n1,a_min=0, a_max=200):

        prefix = '../../data/RFI_data_Jul_2019/train_800_time_smear_'
        
#        train_800_time_smear_dirty_vis_amp.h5
#         train_800_time_smear_dirty_vis_amp
#        train_800_time_smear_rfi_vis_amp.h5
#        train_800_time_smear_dirty_vis_phs.h5
#        train_800_time_smear_rfi_vis_phs.h5    
        fname = prefix+'dirty_vis_phs.h5'
        self.X = read_h5file(fname,dataset=u''+fname[:-3])
        fname = prefix+'rfi_vis_amp.h5'
        self.Y = read_h5file(fname,dataset=u''+fname[:-3])

#        self.X = np.clip(np.fabs(self.X), a_min, a_max)
#        self.Y = np.clip(np.fabs(self.Y), a_min, a_max)
        self.nx = nx
        self.ny = ny
        self.n0 = n0
        self.n1 = n1
    
    def __call__(self,n):
        xp = []
        yp = []
        for i in range(n):
            nimg = np.random.randint(self.n0,self.n1)
            xx,yy = get_slice(self.X[nimg],self.Y[nimg],self.nx,self.ny)
            xp.append(xx)
            yp.append(yy)
     
        xp, yp = np.array(xp), np.array(yp)
#        xp, yp = np.expand_dims(xp,-1), np.expand_dims(yp,-1)
#        print(xp.shape,yp.shape)
#        exit()
        return xp, yp
    
    def getter(self,nimg):
        xp, yp = self.X[nimg],self.Y[nimg]
        xp, yp = np.array(xp), np.array(yp)
        xp, yp = np.expand_dims(xp,-1), np.expand_dims(yp,-1)
        xp, yp = np.expand_dims(xp,0), np.expand_dims(yp,0)
        return xp, yp


class Data_Provider_Jul_2019_2Channel(object):

    def __init__(self,nx,ny,n0,n1,a_min=0, a_max=200):

        prefix = '../../data/RFI_data_Jul_2019/train_800_time_smear_'
        
#        train_800_time_smear_dirty_vis_amp.h5
#         train_800_time_smear_dirty_vis_amp
#        train_800_time_smear_rfi_vis_amp.h5
#        train_800_time_smear_dirty_vis_phs.h5
#        train_800_time_smear_rfi_vis_phs.h5    
        fname = prefix+'dirty_vis_amp.h5'
        amp = read_h5file(fname,dataset=u''+fname[:-3])
        fname = prefix+'dirty_vis_phs.h5'
        phs = read_h5file(fname,dataset=u''+fname[:-3])
        self.X = np.concatenate([amp,phs],axis=-1)
        
        fname = prefix+'rfi_vis_amp.h5'
        self.Y = read_h5file(fname,dataset=u''+fname[:-3])

#        self.X = np.clip(np.fabs(self.X), a_min, a_max)
#        self.Y = np.clip(np.fabs(self.Y), a_min, a_max)
        self.nx = nx
        self.ny = ny
        self.n0 = n0
        self.n1 = n1
    
    def __call__(self,n):
        xp = []
        yp = []
        for i in range(n):
            nimg = np.random.randint(self.n0,self.n1)
            xx,yy = get_slice(self.X[nimg],self.Y[nimg],self.nx,self.ny)
            xp.append(xx)
            yp.append(yy)
     
        xp, yp = np.array(xp), np.array(yp)
#        xp, yp = np.expand_dims(xp,-1), np.expand_dims(yp,-1)
#        print(xp.shape,yp.shape)
#        exit()
        return xp, yp
    
    def getter(self,nimg):
        xp, yp = self.X[nimg],self.Y[nimg]
        xp, yp = np.array(xp), np.array(yp)
        xp, yp = np.expand_dims(xp,-1), np.expand_dims(yp,-1)
        xp, yp = np.expand_dims(xp,0), np.expand_dims(yp,0)
        return xp, yp

def the_print(text,style='bold',tc='gray',bgc='red'):
    """
    prints table of formatted text format options
    """
    colors = ['black','red','green','yellow','blue','purple','skyblue','gray']
    if style == 'bold':
        style = 1
    elif style == 'underlined':
        style = 4
    else:
        style = 0
    fg = 30+colors.index(tc)
    bg = 40+colors.index(bgc)
    
    form = ';'.join([str(style), str(fg), str(bg)])
    print('\x1b[%sm %s \x1b[0m' % (form, text))

def ch_mkdir(directory):
    """
    ch_mkdir : This function creates a directory if it does not exist.

    Arguments:
        directory (string): Path to the directory.

    --------
    Returns:
        null.		
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except:
            print('could not make the directory!')



class Test_Provider_Jul_2019(object):

    def __init__(self,chan):

        prefix = '../../data/RFI_data_Jul_2019/test_800_time_smear_'
        fname = prefix+'dirty_vis_phs.h5'
        amp = read_h5file(fname,dataset=u''+fname[:-3])
        fname = prefix+'dirty_vis_amp.h5'
        phs = read_h5file(fname,dataset=u''+fname[:-3])
        if chan!=0:
            self.X = np.concatenate([amp,phs],axis=-1)
        else:
            self.X = amp#np.expand_dims(self.X,-1)
        self.num = len(self.X)
        self.num, self.nx, self.ny, self.n_chan = self.X.shape
           
        fname = prefix+'rfi_vis_amp.h5'
        self.Y = read_h5file(fname,dataset=u''+fname[:-3])
        self.i = 0
    
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.num:
            xp, yp = self.X[self.i],self.Y[self.i]
            self.i += 1
            xp, yp = np.expand_dims(xp,0), np.expand_dims(yp,0)
            return xp, yp
        else:
            raise StopIteration
            
    def next(self):
        if self.i < self.num:
            xp, yp = self.X[self.i],self.Y[self.i]
            self.i += 1
            xp, yp = np.expand_dims(xp,0), np.expand_dims(yp,0)
            return xp, yp
        else:
            return None,None











#def get_slice2D(x,y,nx,ny):
#    lx,ly = x.shape[:2]
#    if nx==0 or nx==lx:
#        slx = slice(0, lx)                
#    else:
#        idx = np.random.randint(0, lx - nx)            
#        slx = slice(idx, (idx+nx))       
#    if ny==0 or ny==ly:
#        sly = slice(0, ly)                
#    else:
#        idy = np.random.randint(0, ly - ny)            
#        sly = slice(idy, (idy+ny))
#    return x[slx, sly],y[slx, sly]

#def get_slice3D(x,y,nx,ny,nz):
#    lx,ly,lz = x.shape[:3]
#    if nx==0 or nx==lx:
#        slx = slice(0, lx)                
#    else:
#        idx = np.random.randint(0, lx - nx)            
#        slx = slice(idx, (idx+nx))       
#    if ny==0 or ny==ly:
#        sly = slice(0, ly)                
#    else:
#        idy = np.random.randint(0, ly - ny)            
#        sly = slice(idy, (idy+ny))
#    if nz==0 or nz==lz:
#        slz = slice(0, lz)                
#    else:
#        idz = np.random.randint(0, lz - nz)            
#        slz = slice(idz, (idz+nz))
#    return x[slx, sly, slz],y[slx, sly, slz]
#train_splt,test_splt,valid_splt = .4,.4,.2
def get_slice(x,window):
    if window is None:
        window = x.ndim*[0]
    window = np.array(window)
    shape = np.array(x.shape)
    idx0 = np.random.randint(shape-window)
    slices = []
    for i in range(x.ndim):
        if window[i]!=0:
            slices.append(slice(idx0[i],idx0[i]+window[i]))
        else:
            slices.append(slice(0,shape[i]))
    return slices

def slicer(x,slices):
    slcstr = ','.join(['slices[{}]'.format(i) for i in range(len(slices))])
    xp = eval('x[{}]'.format(slcstr))
    return xp
    
def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

class Data_Provider_Sep_2019(object):

    def __init__(self,nx,ny,prefix,
                      process,
                      window = None,
                      n_batch=5,
                      call_freq_train = 10,
                      call_freq_valid = 10000,
                      split = [0.4,0.2,0.4],
                      phase_in = False,
                      phase_out = False,
                      postprocess = None):
    
#        self.clean_list = sorted(glob(prefix+'clean_abs_*.h5'))
#        self.dirrty_list = sorted(glob(prefix+'dirty_abs_*.h5'))

        self.clean_list = [prefix+'clean_'+str(i)+'.h5' for i in range(100)]
        self.dirty_list = [prefix+'dirty_'+str(i)+'.h5' for i in range(100)]
        
        n_clean = len(self.clean_list)
        n_dirty = len(self.dirty_list)
        
        assert n_clean==n_dirty, 'Clean/Dirty files are not equal!'
        self.n_data = n_clean
        
        
        self.window = window
        self.n_batch = n_batch
        self.prefix = prefix
        self.phase_in = phase_in
        self.phase_out = phase_out
        self.process = process
        self.postprocess = postprocess
        
#        (100, 15, 4096, 4)
        x = read_h5file(self.clean_list[0])
        self.n_time, self.n_baseline, self.n_freq, self.n_pol = x.shape

        inds = np.arange(self.n_data)
        if len(split)==3 and split[0]<1:
            if np.sum(split)!=1:
                print('The split sum have to be 1. Test set will be changed.')
                split[2] = 1-split[0]-split[1]
            n1 = int(split[0]*self.n_data)
            n2 = int((split[0]+split[1])*self.n_data)
        elif len(split)==2 and split[0]>=2:
            n1 = int(split[0])
            n2 = int(split[1])      
        else:
            print('unknown splitting configuration, the defualt [0.4,0.2,0.4] will be used.')
            split = [0.4,0.2,0.4]
            n1 = int(split[0]*self.n_data)
            n2 = int((split[0]+split[1])*self.n_data)
            
        self.inds_train = inds[:n1]
        self.inds_valid = inds[n1:n2]
        self.inds_test  = inds[n2:]
        
        print(self.inds_train)
        print(self.inds_valid)
        print(self.inds_test)
        
        self.call_freq_train = call_freq_train
        self.call_freq_valid = call_freq_valid
        
        self.traincall = 0
        self.validcall = 0
        self.testcall = 0
        self.load_time = 0
        self.train_call_time = 0
        self.valid_call_time = 0

    def reload(self,inds,alpha = None):
        t0 = time()
        
        indsr = inds+0
#        np.random.seed()
#        np.random.shuffle(indsr)
        shuffle(indsr)
        indsr = indsr[:self.n_batch]
        print(indsr)
        
        X = []
        Y = []
        for i in indsr:      
            clean = read_h5file(self.clean_list[i])
            dirty = read_h5file(self.dirty_list[i])
            sigma = np.random.uniform(0.168,2*0.168)
            if alpha is None:
                alpha = 2**np.random.uniform(-10,0)
#                alpha = 2**np.random.uniform(-1,1)
            
#            noise = np.random.normal(0,sigma,clean.shape)
            noise = complex_noise(clean,0,sigma)
            rfi = alpha * dirty
            dirty = clean + rfi + noise
            
            if not self.phase_in:
                dirty = np.abs(dirty).astype(np.float32)
            else:
                dirty = np.stack([np.abs(dirty).astype(np.float32),
                                  np.angle(dirty).astype(np.float32)],axis=-1)
            
            if not self.phase_out:
                rfi = np.abs(rfi).astype(np.float32)
            else:
                rfi = np.stack([np.abs(rfi).astype(np.float32),
                                np.angle(rfi).astype(np.float32)],axis=-1)

            X.append(dirty)
            Y.append(rfi)
        X,Y = np.array(X),np.array(Y)
        X,Y = self.process(X,Y)
        
        if self.postprocess is not None:
            X,Y = self.postprocess(X,Y)
        
        if self.load_time==0:
            self.load_time = (time()-t0)/len(inds)
        else:
            self.load_time = np.mean([self.load_time,(time()-t0)/len(inds)])
        return X,Y
    
    def get_train(self,n):
        if self.traincall % self.call_freq_train==0:
            self.X_train,self.Y_train = self.reload(self.inds_train)
            self.n_train = len(self.X_train)
#            print(self.X_train.shape,self.Y_train.shape)
        t0 = time()
        inds = np.arange(self.n_train)
        np.random.shuffle(inds)
        inds = inds[:n]
        x,y = self.X_train[inds],self.Y_train[inds]

        slices = get_slice(x,self.window)
        x = slicer(x,slices)
        y = slicer(y,slices)
       
        self.traincall = self.traincall+1
        self.train_call_time += time()-t0
        
        return x,y

    def get_valid(self):
        if self.validcall % self.call_freq_valid==0:
            self.X_valid,self.Y_valid = self.reload(self.inds_valid)
            self.n_valid = len(self.X_valid)
#            print(self.X_valid.shape,self.Y_valid.shape)
        t0 = time()            
        inds = np.arange(self.n_valid)
        np.random.shuffle(inds)
        inds = inds[:self.n_batch]
        x,y = self.X_valid[inds],self.Y_valid[inds]        
        
        self.validcall = self.validcall+1
        self.valid_call_time += time()-t0
        
        return x,y

    def get_test(self,alpha=None):
        i = self.inds_test[self.testcall]
        print('Alpha is',alpha)
        X_test,Y_test = self.reload(np.array([i]), alpha=alpha)
#        print(X_test.shape,Y_test.shape)
        self.testcall = self.testcall+1
        
        return X_test,Y_test
        
    def __iter__(self):
        self.testcall = 0
        return self
        
    def __next__(self):
        try:
            return self.get_test()
        except IndexError:
            raise StopIteration
        except:
            print('Unknown erro in iteration!')
    

    def test_reset(self):
        self.testcall = 0

    def report(self):
        print('Average load time is {:1.1f} sec'.format(self.load_time))
        s = getsizeof(self.X_train+0)
        print('Size of X_train is {}'.format(sizeof_fmt(s)))
        s = getsizeof(self.Y_train+0)
        print('Size of Y_train is {}'.format(sizeof_fmt(s)))
        s = getsizeof(self.X_valid+0)
        print('Size of X_valid is {}'.format(sizeof_fmt(s)))
        s = getsizeof(self.Y_valid+0)
        print('Size of Y_valid is {}'.format(sizeof_fmt(s)))  
        tct = self.train_call_time/self.traincall
        print('Average train time call is {:1.1f} sec'.format(tct))
        vct = self.valid_call_time/self.validcall
        print('Average valid time call is {:1.1f} sec'.format(vct))
        vol = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print('Maximum memory usage is {}'.format(sizeof_fmt(1024*vol)))


def process(X,Y,mode,phase_in,phase_out):
    if phase_in:
        nch_x = 2
    else:
        nch_x = 1
        
    if phase_out:
        nch_y = 2
    else:
        nch_y = 1
# (5, 100, 15, 4096, 4)

    if mode==1:
        X,Y = np.swapaxes(X,1,4),np.swapaxes(Y,1,4)
        X,Y = np.swapaxes(X,1,2),np.swapaxes(Y,1,2)
        X,Y = X.reshape(-1,4096,100,1*nch_x),Y.reshape(-1,4096,100,1*nch_y)
#    (300, 4096, 100, 1)
    if mode==2:
        X,Y = np.swapaxes(X,1,2),np.swapaxes(Y,1,2)
        X,Y = np.swapaxes(X,2,3),np.swapaxes(Y,2,3)
        X,Y = X.reshape(-1,4096,100,4*nch_x),Y.reshape(-1,4096,100,4*nch_y)
#    (75, 4096, 100, 4)
    if mode==3:
        X,Y = np.swapaxes(X,2,4),np.swapaxes(Y,2,4)
        X,Y = np.swapaxes(X,1,2),np.swapaxes(Y,1,2)
        X,Y = np.swapaxes(X,2,3),np.swapaxes(Y,2,3)
        X,Y = X.reshape(-1,4096,100,15*nch_x),Y.reshape(-1,4096,100,15*nch_y)
#    (20, 4096, 100, 15)
    if mode==4:
        X,Y = np.swapaxes(X,1,2),np.swapaxes(Y,1,2)
        X,Y = np.swapaxes(X,2,3),np.swapaxes(Y,2,3)
#    (5, 15, 4096, 100, 4)
    
    return X,Y


