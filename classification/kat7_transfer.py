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
from random import shuffle

from utils import *
from sklearn.metrics import matthews_corrcoef
from mydataprovider import UnetDataProvider as DataProvider

parser = argparse.ArgumentParser()
parser.add_argument('--arch', required=False, help='choose architecture', type=str, default='1')
parser.add_argument('--ws', required=False, help='time window', type=int, default=200)
parser.add_argument('--time_limit', action="store", type=int, default=60)
parser.add_argument('--learning_rate', action="store", type=float, default=0.05)
parser.add_argument('--train', action="store_true", default=False)
parser.add_argument('--test', action="store_true")
parser.add_argument('--nqq', required=False, help='number of Q', type=int, default=1)

parser.add_argument('--ntry', required=True, help='choose architecture', type=int)
parser.add_argument('--shuffle', action="store_true")

args = parser.parse_args()
arch = args.arch
ws = args.ws
time_limit = args.time_limit
learning_rate = args.learning_rate
nqq = args.nqq
ntry = args.ntry # number of the fold




if ntry>3:
    print('Too large ntry')
    exit()

dataset_pres = []

for ant1 in range(0,6):
    for ant2 in range(ant1+1,7):
        for pol in range(4):
            dataset_pres.append('ant1={}-ant2={}-pol={}.h5'.format(ant1,ant2,pol))

n_data = len(dataset_pres)

indx_tot = np.arange(n_data)
test_portion = ((n_data-4)//4) # excluding one whole polarization since 21%4=1
indx_test = np.arange(ntry*test_portion,(ntry+1)*test_portion)
indx_train = np.setdiff1d(indx_tot,indx_test)

dataset_pres = np.array(dataset_pres)
if args.shuffle:
    shuffle(dataset_pres)  
print(dataset_pres[indx_test])
print(dataset_pres[indx_train])

prefix = '../../data/alireza/1375091767.full_pol/fulldata/h5/dataset/'
#prefix = '../../data/alireza/1375091767.full_pol/h5/dataset/'

train_files = []
for i in dataset_pres[indx_train[:]]:
    train_files.append(prefix+i)

test_files = []
for i in dataset_pres[indx_test]:
    test_files.append(prefix+i)

#files_list = sorted(glob.glob('../../data/alireza/1375091767.full_pol/h5/train/*.h5'))
#test_files = sorted(glob.glob('../../data/alireza/1375091767.full_pol/h5/test/*.h5'))



the_print('number of training files: '+str(len(train_files)))

#if ntry=='1':
#    nnn = 5
#elif ntry=='2':
#else:
#    nnn = 1
#else:
#    print('WTF!')
#    exit()
     
threshold = 0

dpt = DataProvider(nx=ws,
                   files=train_files,
                   threshold=threshold,
                   sim='kat7',
                   r_freq=500,
                   n_loaded=10,
                   a_min=0,
                   a_max=200,
                   n_class=1)

_,nx,ny,_ = dpt(1)[0].shape
print(nx,ny)

the_print('Chosen architecture is: '+args.arch+'; learning rate = '+str(learning_rate),bgc='green')

name = arch+'_'+str(nx)+'x'+str(ny)+'_try'+str(ntry)
if args.shuffle:
    name = name+'_sh'
print(name)
model_add = './models/trkat7_model_'+name
pred_dir = 'predictions/trkat7_model_'+name+'/'
ch_mkdir(pred_dir)
res_file = 'results/trkat7_model_'+name
ch_mkdir('results')
model0_add = 'models/mk_model_'+arch+'_4096x50_try2'

qq0 = 0
decay = 1.02
if isfile(res_file+'_prop.npy'):
    try:
        qq0,learning_rate = np.load(res_file+'_prop.npy')
    except:
        qq0 = np.load(res_file+'_prop.npy')
        learning_rate = learning_rate/(decay)**qq0
    qq0 = int(qq0)
    qq0 = qq0+1

#qq0 = 0
#learning_rate = 0.05
print('The learning will begin from {} q number.'.format(qq0))
#exit()
ns = 10
#learning_rate = 5e-6
if qq0==0:
    model_add_now = model0_add
else:
    model_add_now = model_add

exec('from arch_'+arch+' import architecture')
def arch(x_in):
    return architecture(x_in, trainable=0)

restore = 1
conv = ng.Model(data_provider=dpt,
             restore=restore,
             model_add=model_add_now,
             arch=arch)

print(conv.model_add)
conv.model_add = model_add
print(conv.model_add)
print('')
print('Number of trainable variables: {:d}'.format(conv.n_variables))
#nqq = qq0+1
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
               resuscitation_limit=150)
    
    learning_rate = learning_rate/decay

    prop = np.array([qq,learning_rate])
    np.save(res_file+'_prop',prop)   
    

#    import pickle
#    weights = conv.get_filters()
#    with open(res_file+'_filters', 'w') as filehandler:
#        pickle.dump(weights, filehandler)
#    np.save(res_file+'_filters',weights)

if args.test:
    times = []
    pr_list = []
    roc_list = []
    auc_list = []
    mcc_list = []
    
    clrs = ['b','r','g','k']
    trsh = np.linspace(1,0,100,endpoint=1)

    for fil in test_files:

        fname = fil.split('/')[-1]
        
        dp = DataProvider(nx=0,
                  files=[fil],
                  threshold=threshold,
                  sim='kat7',
                  r_freq=5,
                  n_loaded=1,
                  a_min=0,
                  a_max=200,
                  n_class=1)
        
        data,mask = dp(1)
        
        pred = conv.conv_large_image(data,pad=10,lx=nx,ly=ny)
        
        mask = mask[0,8:-8,:,0]
        pred = pred[0,8:-8,:,0]

        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,8))
        ax1.imshow(data[0,:,:,0],aspect='auto')
        ax2.imshow(mask,aspect='auto')
        ax3.imshow(pred,aspect='auto')

        np.save(pred_dir+fname+'_mask',mask)
        np.save(pred_dir+fname+'_pred',pred)

        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'.jpg',dpi=30)
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
        
        
        
#        dp = DataProvider(nx=0,a_min=0, a_max=200, files=[fil], n_class=1)
#        data0,mask0 = dp(1)
#        
#        nt = data0.shape[2]
#        chunks = np.array_split(np.arange(nt).astype(int), 3)
#        idns = np.arange(3)
#        np.random.shuffle(idns)
#        chunk_list = np.array(chunks)[idns[:2]]
#        
#        if qq==qtest:
#            chunk_list = np.array(chunks)
#        
#        for ind in chunk_list:
#            data,mask = data0[:,:,ind,:],mask0[:,:,ind,:]
#            pred = conv.conv_large_image(data,pad=10,lx=nx,ly=ny)
#        
#            fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,8))
#            ax1.imshow(data[0,:,:,0],aspect='auto')
#            ax2.imshow(mask[0,:,:,0],aspect='auto')
#            ax3.imshow(pred[0,:,:,0],aspect='auto')
#            
#            mask = mask[0,10:-10,10:-10,0]
#            pred = pred[0,10:-10,10:-10,0]
#            np.save(pred_dir+fname+'_'+str(qq)+'_mask',mask)
#            np.save(pred_dir+fname+'_'+str(qq)+'_pred',pred)

#            plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
#            plt.savefig(pred_dir+fname+'_'+str(qq)+'.jpg',dpi=30)
#            plt.close()

#            y_true = mask.reshape(-1).astype(int)
#            y_score = pred.reshape(-1)
#            y_score /= y_score.max()

#            recall,precision = prc(y_true, y_score, trsh)
#            pr_list.append(np.stack([recall,precision]).T)
#            
#            fpr,tpr = rocc(y_true, y_score, trsh)
#            roc_list.append(np.stack([fpr,tpr]).T)    
#            auc_list.append(np.trapz(tpr, fpr))
#            
#    np.save(res_file+'_'+str(qq)+'_pr',np.array(pr_list))
#    np.save(res_file+'_'+str(qq)+'_roc',np.array(roc_list))   
#    print(np.mean(auc_list))
    

