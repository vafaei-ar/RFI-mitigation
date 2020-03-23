import os
import time
import numpy as np
from skimage import draw
from skimage import measure
from astropy.io import fits
from astropy import units as u
from astropy import wcs, coordinates
from scipy.ndimage.filters import gaussian_filter

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
    
def tfpnr_trsh(truth, prob, trsh):
    pred = prob>=trsh

    return tfpnr(truth, pred)

def rocc(truth, pred, trsh):
    trshp = np.concatenate(([1.+eps], trsh, [-eps]))
    pred = pred-pred.min()
    pred = pred/pred.max()
    tpr = []
    fpr = []
    for tr in trshp:
        tp,fp,tn,fn = tfpnr_trsh(truth, pred, tr)
        if tp+fn!=0:
            tpr.append(1.*tp/(tp+fn))
        else:
            tpr.append(1.)
        if fp+tn!=0:
            fpr.append(1.*fp/(fp+tn))
        else:
            fpr.append(1.)
    return fpr,tpr

def prc(truth, pred, trsh):
    trshp = np.concatenate(([1.+eps], trsh, [-eps]))
    pred = pred-pred.min()
    pred = pred/pred.max()
    recall = []
    precision = []
    for tr in trshp:
        tp,fp,tn,fn = tfpnr_trsh(truth, pred, tr)
        if tp+fn!=0:
            recall.append(1.*tp/(tp+fn))
        else:
            recall.append(1.)
        if tp+fp!=0:
            precision.append(1.*tp/(tp+fp))
        else:
            precision.append(1.)
    return recall,precision

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

def standard(X):
	"""
	standard : This function makes data ragbe between 0 and 1.
	
	Arguments:
		X (numoy array) : input data.
	
	--------
	Returns:
		standard data.
	"""
	xmin = X.min()
	X = X-xmin
	xmax = X.max()
	X = X/xmax
	return X

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
		  os.makedirs(directory)

class StopWatch(object):
    
    def __init__(self):
        self.units = [1,7,24,60,60]
        self.unit_names = ['Weeks','Days','Hours','Minutes','Seconds'] 
        self.n_units = len(self.unit_names)
        self.start = self.time2array(time.time())
        
    def reset(self,value=None):
        if value is None:
            self.start = self.time2array(time.time())
        else:
            self.start = self.time2array(time.time())-value
        
    def interval(self):
        return self.calib(self.time2array(time.time())-self.start)
    
    def time2array(self,value):
        '''From seconds to Weeks;Days;Hours:Minutes;Seconds'''
        value = int(value)
#         seconds = value%60
#         value = value//60
#         minutes = value%60
#         value = value//60
#         hours = value%24
#         value = value//24
#         days = value%7
#         weeks = value//7
#         return np.array([weeks,days,hours,minutes,seconds])
        res = []
        for un in self.units[::-1]:
            res.append(value%un)
            value = int(value/un)
        return np.array(res[::-1])

    def __str__(self):
        res = self.interval()
        dont = True
        string = ''
        values = []
        for i in range(self.n_units):
            if res[i]==0 and dont:
                continue
            else:
                dont = False
                string += '{} '+self.unit_names[i]+'; '
                values.append(res[i])
        string = string[:-2]
        return string.format(*values)
        
    def __repr__(self):
        res = self.interval()
        dont = True
        string = ''
        values = []
        for i in range(self.n_units):
            if res[i]==0 and dont:
                continue
            else:
                dont = False
                string += '{} '+self.unit_names[i]+'; '
                values.append(res[i])
        string = string[:-2]
        return string.format(*values)
    
    def __call__(self):
        return self.interval()   
    
    def calib(self,res):
        for i in range(self.n_units-1):
            ii = self.n_units-i-1
            while res[ii]<0:
                res[ii-1] = res[ii-1]-1
                res[ii] = res[ii]+self.units[ii]
            res[ii-1] = res[ii-1]+res[ii]//self.units[ii]
            res[ii] = res[ii]%self.units[ii]
        return res
    
    def _print(self,res):
        dont = True
        res = self.calib(res)
        string = ''
        values = []
        for i in range(self.n_units):
            if res[i]==0 and dont:
                continue
            else:
#                 dont = False
                string += '{} '+self.unit_names[i]+'; '
                values.append(res[i])
        string = string[:-2]
        print(string.format(*values))   
        
        
#from sklearn.metrics import roc_curve, auc,precision_recall_curve

#def tfpnr(truth, pred, trsh):
#    pred = pred-pred.min()
#    pred = pred/pred.max()
#    
#    pred_mask = pred>trsh
#    
#    pos = 1.*np.sum(pred_mask)
#    neg = 1.*np.sum(~pred_mask)
#    
#    tp = np.sum(truth[pred_mask]==True)
##     fp = np.sum(truth[pred_mask]==False)
#    fp = pos-tp
#    tn = np.sum(truth[~pred_mask]==False)
##     fn = np.sum(truth[~pred_mask]==True)
#    fn = neg-tn

#    return tp,fp,tn,fn

#def rocc(truth, pred, trsh):
#    tpr = []
#    fpr = []
#    for tr in trsh:
#        tp,fp,tn,fn = tfpnr(truth, pred, tr)
#        tpr.append(1.*tp/(tp+fn))
#        fpr.append(1.*fp/(fp+tn))
#    return fpr,tpr

#def prc(truth, pred, trsh):
#    recall = []
#    precision = []
#    for tr in trsh:
#        tp,fp,tn,fn = tfpnr(truth, pred, tr)
#        recall.append(1.*tp/(tp+fn))
#        precision.append(1.*tp/(tp+fp))
#    return recall,precision


#def ps_extract(xp):
#	xp = xp-xp.min() 
#	xp = xp/xp.max()

#	nb = []
#	for trsh in np.linspace(0,0.2,200):
#		  blobs = measure.label(xp>trsh)
#		  nn = np.unique(blobs).shape[0]
#		  nb.append(nn)
#	nb = np.array(nb)
#	nb = np.diff(nb)
#	trshs = np.linspace(0,0.2,200)[:-1]
#	thrsl = trshs[~((-5<nb) & (nb<5))]
#	if thrsl.shape[0]==0:
#		trsh = 0.1
#	else:
#		trsh = thrsl[-1]
#2: 15, 20
#3: 30,10
#4: 50, 10
#	nnp = 0
#	for tr in np.linspace(1,0,1000):
#		blobs = measure.label(xp>tr)
#		nn = np.unique(blobs).shape[0]
#		if nn-nnp>50:
#				break
#		nnp = nn
#		trsh = tr

#	blobs = measure.label(xp>trsh)
#	xl = []
#	yl = []
#	pl = []
#	for v in np.unique(blobs)[1:]:
#		filt = blobs==v
#		pnt = np.round(np.mean(np.argwhere(filt),axis=0)).astype(int)
#		if filt.sum()>10:
#			xl.append(pnt[1])
#			yl.append(pnt[0])
#			pl.append(np.mean(xp[blobs==v]))
#	return np.array([xl,yl]).T,np.array(pl)
