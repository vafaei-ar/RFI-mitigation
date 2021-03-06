# RFI-classification

This directory containes the developed scripts for RFI classification. The pipe line is demotestrated in the bellow figure:

<p align="center">
  <img src="../images/pipeline.jpg" width="700"/>
</p>

A sample of the results on HIDE simulations are:

<p align="center">
  <img src="../images/samples.jpg" width="600"/>
</p>

The files are as follows:

**arch_?.py**s are the architectures. 

**mydataprovider.py** contains the data provider we used to feed the CNN. Pleaase notice that the data provider works in different modes based on the simulations we have. You may want to develope your own data provider.

**utils.py** contains some auxiliary fucntions.

The rest of the files are scripts to train either R-net or U-net on HIDE, MeerKAT and KAT7 datasets. 

The **trained models** will be shared after acceptance.


**Citing RFI-classification:** 
```  
@article{vafaei2020deep,
  title={Deep learning improves identification of Radio Frequency Interference},
  author={Vafaei Sadr, Alireza and Bassett, Bruce A and Oozeer, Nadeem and Fantaye, Yabebal and Finlay, Chris},
  journal={Monthly Notices of the Royal Astronomical Society},
  volume={499},
  number={1},
  pages={379--390},
  year={2020},
  publisher={Oxford University Press}
}
```




