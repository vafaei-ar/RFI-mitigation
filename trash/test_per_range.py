import os
import argparse
import numpy as np
#from utils import *
from glob import glob


parser = argparse.ArgumentParser()
#parser.add_argument('--ws', required=False, help='time window', type=int, default=100)
parser.add_argument('--mode', required=False, help='theme', type=int, default=1)
parser.add_argument('--w1', required=False, help='nt', type=int, default=4000)
parser.add_argument('--w2', required=False, help='nt', type=int, default=50)
parser.add_argument('--ntry', required=True, help='nt', type=str)

parser.add_argument('--pin', required=False, help='theme', type=int, default=0)
parser.add_argument('--pout', required=False, help='theme', type=int, default=0)
parser.add_argument('--sep', required=False, help='theme', type=int, default=0)
parser.add_argument('--app', required=False, help='theme', type=int, default=1)

args = parser.parse_args()
mode = args.mode
w1 = args.w1
w2 = args.w2
n_try = args.ntry
phase_in = args.pin
phase_out = args.pout
sep = args.sep
app = args.app

if sep==0:
    p2 = None
elif sep and (mode==2 or mode==3):
    def p2(X,Y):
        return X[...,sep-1:sep],Y[...,sep-1:sep]
else:
    print('Seperation is only available for polarization!')
    exit()

name = ''
pfix = 'try'+n_try+'_'
if phase_in:
    pfix += 'in_'
if phase_out:
    pfix += 'out_'
if sep:
    pfix += str(sep)+'_'
if app!=1:
    pfix += 'app'+str(app)+'_'
model_add = './models/'+pfix+str(mode)#+'_'+name

files = glob(model_add+'/preds/*.npz')
n_files = len(files)

minlog = np.log(1e-25) #1e-21
maxlog = np.log(1e10) #1e8
rngs = np.linspace(minlog,maxlog,100)
rngs = np.insert(rngs,0,-200)
loss = np.zeros((1,100,n_files))
num = np.zeros((1,100,n_files))
losst = np.zeros((1,n_files))

n_in = np.zeros((1,n_files))

for i in range(n_files):

    d = np.load(files[i])
    truth = d['truth']
    pred = d['pred']
    n_in[0,i] = pred.shape[0]
    
#    print(files[i].split('/')[-1][:-4])
#    print(truth.shape,pred.shape)

#    truth, pred = truth[...,0], pred[...,0]
    losst[0,i] = np.mean(( np.log(truth+1)-np.log(pred+1) )**2)
    truthp = truth+0
    truthp[truthp==0] = 1e-50
    truthp = np.log(truthp)
    
#    print(truth.shape,pred.shape)
    for j in range(100):
        filt = (rngs[j]<=truthp) & (truthp<rngs[j+1])
        truth2 = truth[filt]
        pred2 = pred[filt]
        loss[0,j,i] = np.sum(( np.log(truth2+1)-np.log(pred2+1) )**2)
        num[0,j,i] = np.sum(filt)
        
np.savez(model_add+'/test_res',loss = loss, num=num, losst = losst, n_in = n_in)
##        
        
        
        
        
        
        
        
        
        

























