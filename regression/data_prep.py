from glob import glob
import h5py as h5
import numpy as np
#from sklearn.utils import shuffle

#def open_h5file(filename):
#    #function to read HDF5 file format
#    return h5.File(filename,'r')

#def read_h5file(filename,dataset='dataset'):
#    #function to read a hdf5 file and output the array as a np.array
#    data = open_h5file(filename)
#    return data[dataset][()]

#def save_h5file(savename,input_array,dataset='dataset'):
#    #function to save hdf5 
#    out = h5.File(savename, 'w')
#    out.create_dataset(dataset, data=input_array)
#    out.close()

#def extract_complex_vis(filename):
#    #function to extract complex dirty and clean visibility from chris sim data
#    data = open_h5file(filename)
#    group = data['output']
#    group_clean = group['vis_clean']
#    group_dirty = group['vis_dirty']
#    return group_clean[()],group_dirty[()]

#def smear(data,axis=0,check=0):
#    n_time = data.shape[axis]
#    nt = int(n_time//8)
#    sdata = np.array(np.split(data, nt, axis=axis))
#    #print(sdata.shape)

#    ## TO CHECK THE SMEARING WORKS RIGHT.
#    if check:
#        for i in range(nt):
#            dd = np.all(sdata[i,:,0,0,0]==data[i*8:i*8+8,0,0,0])
#            if not dd:
#                print(i)
#                
#    return np.mean(sdata,axis=axis+1)
#    
#def complex_noise(arr_in,mu,sigma,seed=0):
#    #Function to create real and imaginary noise for an array.
#    np.random.seed(seed)
#    real_part = np.random.normal(mu,sigma,arr_in.shape)
#    im_part = np.random.normal(mu,sigma,arr_in.shape)
#    noise = real_part + 1j*im_part
#    return noise
    
prefix = '../../data/RFI_data_Sep_2019/'
baselines = 15
pol = 4
#num_of_100_time_steps = 8
reduced_frame_num = baselines*pol#*num_of_100_time_steps

#from sys import argv
#iii = int(argv[1])
#[iii:iii+10]

list_of_sim_dat = sorted(glob(prefix+'date*.h5'))
n_files = len(list_of_sim_dat)
#print(list_of_sim_dat)


ii = 0
for i in range(n_files):
    filename = list_of_sim_dat[i]
    print(i,end='\r')
    
    try:
        clean,dirty = extract_complex_vis(filename)

    #    print(clean.shape,dirty.shape)
    #    print(clean.dtype,dirty.dtype)

        clean = smear(clean).astype(np.complex64)
        dirty = smear(dirty).astype(np.complex64)

    #    print(clean.shape,dirty.shape)
    #    print(clean.dtype,dirty.dtype)
        
    #    print(clean_abs.dtype,clean_phs.dtype,dirty_abs.dtype,dirty_phs.dtype)
        save_h5file(prefix+'prepared/clean_'+str(ii)+'.h5',clean)
        save_h5file(prefix+'prepared/dirty_'+str(ii)+'.h5',dirty)

        ii += 1

    except:
        print(filename)

#    clean_abs = np.abs(clean).astype(np.float32)
#    clean_phs = np.angle(clean).astype(np.float32)

#    dirty_abs = np.abs(dirty).astype(np.float32)
#    dirty_phs = np.angle(dirty).astype(np.float32)
#    
#    print(clean_abs.dtype,clean_phs.dtype,dirty_abs.dtype,dirty_phs.dtype)
#    save_h5file(prefix+'prepared/clean_abs_'+str(i)+'.h5',clean_abs)
#    save_h5file(prefix+'prepared/clean_phs_'+str(i)+'.h5',clean_phs)
#    save_h5file(prefix+'prepared/dirty_abs_'+str(i)+'.h5',dirty_abs)
#    save_h5file(prefix+'prepared/dirty_phs_'+str(i)+'.h5',dirty_phs)

#alpha = 2**(i-10) for i in [0,1,2,...,10]
#sigma = 0.168*(1+beta) : 0<beta<1

#noise_array = complex_noise(astro,0,0.168)
#dirty_vis = astro + noise_array +  alpha*rfi




