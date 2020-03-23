import matplotlib as mpl
mpl.use('agg')

import os
import glob
import argparse
import numpy as np
import pylab as plt
import ngene as ng
from time import time
from os.path import isfile

from utils import *
from sklearn.metrics import matthews_corrcoef
from mydataprovider import UnetDataProvider as DataProvider
from mydataprovider import read_h5file

parser = argparse.ArgumentParser()
parser.add_argument('--arch', required=False, help='choose architecture', type=str, default='1')
parser.add_argument('--ws', required=False, help='time window', type=int, default=50)
parser.add_argument('--time_limit', action="store", type=int, default=60)
parser.add_argument('--learning_rate', action="store", type=float, default=0.05)
parser.add_argument('--train', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--nqq', required=False, help='number of Q', type=int, default=1)

parser.add_argument('--ntry', required=True, help='choose architecture', type=str)

args = parser.parse_args()
arch = args.arch
ws = args.ws
time_limit = args.time_limit
learning_rate = args.learning_rate
nqq = args.nqq

ntry = args.ntry

prefix = '../../data/RFI_data_Sep_2019/prepared/'
files_list = [prefix+'clean_'+str(i)+'.h5' for i in range(40)]
#test_files = [prefix+'clean_'+str(i)+'.h5' for i in range(40,50)]
the_print('number of training files: '+str(len(files_list)))

if ntry=='1':
    nnn = 5
#elif ntry=='2':
else:
    nnn = 1
#else:
#    print('WTF!')
#    exit()
     
threshold = nnn*0.168

dpt = DataProvider(nx=ws,
                   files=files_list,
                   threshold=threshold,
                   sim='mk',
                   r_freq=500,
                   n_loaded=5,
                   a_min=0,
                   a_max=200,
                   n_class=1)

_,nx,ny,_ = dpt(1)[0].shape
print(nx,ny)

the_print('Chosen architecture is: '+args.arch+'; learning rate = '+str(learning_rate),bgc='green')

name = arch+'_'+str(nx)+'x'+str(ny)+'_try'+ntry
print(name)
model_add = './models/mk_model_'+name
pred_dir = 'predictions/mk_model_'+name+'/'
ch_mkdir(pred_dir)
res_file = 'results/mk_model_'+name
ch_mkdir('results')

qq0 = 0
decay = 1.01

if isfile(res_file+'_prop.npy'):
    try:
        qq0,learning_rate = np.load(res_file+'_prop.npy')
    except:
        qq0 = np.load(res_file+'_prop.npy')
        learning_rate = learning_rate/(decay)**qq0
    qq0 = int(qq0)
    qq0 = qq0+1

print('The learning will begin from {} q number.'.format(qq0))
#exit()
ns = 10
#learning_rate = 5e-6

restore = qq0!=0
conv = ng.Model(data_provider=dpt,
                restore=restore,
                model_add=model_add,
                arch='arch_'+args.arch)
print('')
print('Number of trainable variables: {:d}'.format(conv.n_variables))
nqq = qq0+1
for qq in range(qq0,nqq):
    
    print(qq)
    
    the_print('ROUND: '+str(qq)+', learning rate='+str(learning_rate),bgc='blue')
    conv.train(data_provider=dpt,
               training_epochs = 5,
               iterations=20,
               n_s = ns,
               learning_rate = learning_rate,
               verbose=1,
               death_preliminary_check = 30,
               death_frequency_check = 100,
               resuscitation_limit=150)#, reset=1)
               
    learning_rate = learning_rate/decay
        
    prop = np.array([qq,learning_rate])
    np.save(res_file+'_prop',prop)


#    import pickle
#    weights = conv.get_filters()
#    with open(res_file+'_filters', 'w') as filehandler:
#        pickle.dump(weights, filehandler)
#    np.save(res_file+'_filters',weights)

n_files = 80
prefix = '../../data/RFI_data_Sep_2019/prepared/test_1/'

if args.test:
    pr_list = []
    roc_list = []
    auc_list = []
    mcc_list = []
    
    clrs = ['b','r','g','k']
    trsh = np.linspace(1,0,100,endpoint=1)
    
#    for fil in test_files:

#        print(fil)
#        fname = fil.split('/')[-1]
##        dp = DataProvider(nx=0, a_min=0, a_max=200, files=[fil], n_class=1)        
#        dp = DataProvider(nx=0,
#                          files=[fil],
#                          threshold=0.84,
#                          sim='mk',
#                          r_freq=5,
#                          n_loaded=1,
#                          a_min=0,
#                          a_max=500,
#                          n_class=1)
#        
#        data,mask = dp(1)
    for i in range(n_files):
        fname = str(i)
        data = read_h5file(prefix+'data_'+fname+'.h5')
        rfi = read_h5file(prefix+'rfi_'+fname+'.h5')

        data = np.absolute(data)
        rfi = np.absolute(rfi)
        rfi = rfi>dpt.threshold
        data,mask = dpt._process(data, rfi)

        pred = conv.conv_large_image(data,pad=5,lx=nx,ly=ny)

        mask = mask[0,10:-10,:,0]
        pred = pred[0,10:-10,:,0]

        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,8))
        ax1.imshow(data[0,:,:,0],aspect='auto')
        ax2.imshow(mask,aspect='auto')
        ax3.imshow(pred,aspect='auto')

        np.save(pred_dir+fname+'_mask',mask)
        np.save(pred_dir+fname+'_pred',pred)

        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'.jpg',dpi=70)
        plt.close()

        y_true = mask.reshape(-1).astype(int)
        y_score = pred.reshape(-1)
        y_score /= y_score.max()

        recall,precision = prc(y_true, y_score, trsh)
        pr_list.append(np.stack([recall,precision]).T)
        
        fpr,tpr = rocc(y_true, y_score, trsh)
        roc_list.append(np.stack([fpr,tpr]).T)
        auc_list.append(np.trapz(tpr, fpr))
        
        mcc_list.append(matthews_corrcoef(y_true, y_score.round()))
        
    np.save(res_file+'_pr',np.array(pr_list))
    np.save(res_file+'_roc',np.array(roc_list))
    np.save(res_file+'_mcc',np.array(mcc_list))
    print(np.mean(auc_list),np.mean(mcc_list))



