import matplotlib as mpl
mpl.use('agg')

import sys
import glob
import numpy as np
import argparse
from time import time
import tensorflow as tf 
from os.path import isfile                 
sys.path.insert(0,'../../unet/tf_unet')
import pylab as plt
from sklearn.metrics import matthews_corrcoef
from mydataprovider import UnetDataProvider as DataProvider
from mydataprovider import read_h5file
from tf_unet import unet
from tf_unet import util
#from IPython.display import clear_output

from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--layers', required=False, help='number of layers', type=int, default=3)
parser.add_argument('--feat', required=False, help='choose architecture', type=int, default=32)
parser.add_argument('--ws', required=False, help='time window', type=int, default=50)
parser.add_argument('--nqq', required=False, help='number of Q', type=int, default=1)
#parser.add_argument('--train', action="store_true", default=False)
parser.add_argument('--test', action="store_true", default=False)
parser.add_argument('--ntry', required=True, help='choose architecture', type=str)

args = parser.parse_args()
layers = args.layers
features_root = args.feat
ws = args.ws
nqq = args.nqq

ntry = args.ntry

prefix = '../../data/RFI_data_Sep_2019/prepared/'
files_list = [prefix+'clean_'+str(i)+'.h5' for i in range(40)]
#test_files = [prefix+'clean_'+str(i)+'.h5' for i in range(40,50)]
the_print('number of training files: '+str(len(files_list)))

#dpt = DataProvider(nx=ws,a_min=0, a_max=200, files=files_list)
if ntry=='1':
    nnn = 5
elif ntry=='2':
    nnn = 1
else:
    print('WTF!')
    exit()
     
threshold = nnn*0.168

dpt = DataProvider(nx=ws,
                   files=files_list,
                   threshold=threshold,
                   sim='mk',
                   r_freq=500,
                   n_loaded=5,
                   a_min=0,
                   a_max=200,
                   n_class=2)

_,nx,ny,_ = dpt(1)[0].shape

training_iters = 20
epochs = 5

name = str(layers)+'_'+str(features_root)+'_'+str(nx)+'x'+str(ny)+'_try'+ntry
print(name)
model_dir = './models/mk_unet_'+name
pred_dir = 'predictions/mk_unet_'+name+'/'
ch_mkdir(pred_dir)
res_file = 'results/mk_unet_'+name
ch_mkdir('results')
    
qq0 = 0
if isfile(res_file+'_prop.npy'):
    qq0 = np.load(res_file+'_prop.npy')
    qq0 = int(qq0)
    qq0 = qq0+1

print('The learning will begin from {} q number.'.format(qq0))
nqq = qq0+1
for qq in range(qq0,nqq):
    print(qq)
    restore = qq!=0
    
    tf.reset_default_graph()
    net = unet.Unet(channels=1, 
                    n_class=2, 
                    layers=layers, 
                    features_root=features_root,
                    cost_kwargs=dict(regularizer=0.001))
            
    if not restore:    
        print('')
        n_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Number of trainable variables: {:d}'.format(n_variables))
        
 
    trainer = unet.Trainer(net, batch_size=10, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
    path = trainer.train(dpt, model_dir, 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=0.5, 
                         display_step=1000000,
                         restore=restore,
                         prediction_path = 'prediction/'+name)
                         

    prop = np.array(qq)
    np.save(res_file+'_prop',prop)

n_files = 80
prefix = '../../data/RFI_data_Sep_2019/prepared/test_1/'

if args.test:

    tf.reset_default_graph()
    net = unet.Unet(channels=1, 
                    n_class=2, 
                    layers=layers, 
                    features_root=features_root,
                    cost_kwargs=dict(regularizer=0.001))
 
    trainer = unet.Trainer(net, batch_size=10, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
    path = trainer.train(dpt, model_dir, 
                         training_iters=1, 
                         epochs=1, 
                         dropout=0.5, 
                         display_step=1000000,
                         restore=True,
                         prediction_path = 'prediction/'+name)


    pr_list = []
    roc_list = []
    auc_list = []
    mcc_list = []
    
    clrs = ['b','r','g','k']
    trsh = np.linspace(1,0,100,endpoint=1)
            
    for i in range(n_files):
        fname = str(i)
        data = read_h5file(prefix+'data_'+fname+'.h5')
        rfi = read_h5file(prefix+'rfi_'+fname+'.h5')

        data = np.absolute(data)
        rfi = np.absolute(rfi)
        rfi = (rfi>dpt.threshold)[:,:,:,0]
        print(rfi.shape)
        data,mask = dpt._process(data, rfi)
        print(mask.shape)

        pred,dt = net.predict(model_dir+'/model.cpkt', data,time_it=1)
        _,kk,jj,_, = pred.shape
        mask = util.crop_to_shape(mask, pred.shape)[:,:kk,:jj,:]

        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,8))
        ax1.imshow(data[0,:,:,0],aspect='auto')
        ax2.imshow(mask[0,:,:,1],aspect='auto')
        ax3.imshow(pred[0,:,:,1],aspect='auto')

        np.save(pred_dir+fname+'_mask',mask)
        np.save(pred_dir+fname+'_pred',pred)

        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'.jpg',dpi=30)
        plt.close()

        y_true = mask[0,:,:,1].reshape(-1).astype(int)
        y_score = pred[0,:,:,1].reshape(-1)
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




