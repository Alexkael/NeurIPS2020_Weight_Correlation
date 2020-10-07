version: 
         
         python 3.5+

         tensorflow 1.13.1-gpu
         
         keras 2.2.4
         
WCD method:

The scripts in file_WCD are used to test sgd with and without WCD method.

WCD method in sgwn_CNN.py and sgwn.py.

Results are shown in /WCD/experimental_data/AAA_optimil_loss.ipynb

1..  run: 

         FCN.py                 fully connected network without WCD

         FCN_corr.py            fully connected network with WCD
         
         VGG.py                 VGG without WCD
         
         VGG_corr.py            VGG with WCD, Uncomment sgwn_CNN.py line 87/89/91

2..  

    sgwn_CNN.py: wcd method for CNN

    sgwn.py: wcd method for FCN
    
3..  exchange dataset -> 

                        trn, tst = utils.get_svhn('CNN'/'FCN')

                        trn, tst = utils.get_mnist('CNN'/'FCN')
                        
                        trn, tst = utils.get_fmnist('CNN'/'FCN')
                        
                        trn, tst = utils.get_cifar10('CNN'/'FCN')
    
4..  
   
    initial batch_size -> cfg['SGD_BATCHSIZE'] 

    learning_rate -> cfg['SGD_LEARNINGRATE']

    activitation function -> cfg['ACTIVATION']
    
5.. We train the models with and without WCD **--converge to same-level train loss (training repeatedly, the model with invariant setting may converge to different train loss)--**, then compare the optimal test loss. Thus, we may train the same model (WCD or without WCD) several times to get a model converge to the similar train loss with another one (without WCD or WCD). **--e.g.,--**

         /WCD/experimental_data/                  train loss                                          train loss  
         
         VGG11_corr_cifar10_nol2.out  (converge to)  1.145    VS      VGG11_normal_cifar10_nol2.out    1.140
         
         VGG11_corr_cifar10_nol2_2.out               1.123            VGG11_normal_cifar10_nol2_2.out  1.161
         VGG11_corr_cifar10_nol2_3.out               1.342            VGG11_normal_cifar10_nol2_3.out  1.172
         VGG11_corr_cifar10_nol2_4.out               1.231            VGG11_normal_cifar10_nol2_4.out  1.199
         VGG11_corr_cifar10_nol2_5.out               1.121
                
         AND
         
         VGG16_corr_cifar10_nol2.out  (converge to)  1.015            VGG16_normal_cifar10_nol2.out    1.015
         
         VGG16_corr_cifar10_nol2_2.out               0.995            VGG16_normal_cifar10_nol2_2.out  1.086
         VGG16_corr_cifar10_nol2_3.out               1.098            VGG16_normal_cifar10_nol2_3.out  0.980
         VGG16_corr_cifar10_nol2_4.out               1.045            VGG16_normal_cifar10_nol2_4.out  1.004
                                                                      VGG16_normal_cifar10_nol2_5.out  0.993
                                                                      
         AND
         
         VGG19_corr_cifar10_nol2.out                 1.021            VGG19_normal_cifar10_nol2.out    1.022                
         
         VGG19_corr_cifar10_nol2_2.out               0.987            VGG19_normal_cifar10_nol2_2.out  0.984
         VGG19_corr_cifar10_nol2_3.out               1.014            VGG19_normal_cifar10_nol2_3.out  1.028
         VGG19_corr_cifar10_nol2_4.out               1.010            VGG19_normal_cifar10_nol2_4.out  0.988
         VGG19_corr_cifar10_nol2_5.out               1.056            VGG19_normal_cifar10_nol2_5.out  1.033
         
All experimental data is saved in WCD/experimental_data     


Complexity_Measure:

         Results are shown in Complexity_Measure/complexity_measure_cifar10.ipynb and complexity_measure_cifar100.ipynb

         The raw models for complexity measure are saved in https://drive.google.com/drive/folders/1yfzvu-eQVntjTq_arabLBgGV2Ems98wQ?usp=sharing
