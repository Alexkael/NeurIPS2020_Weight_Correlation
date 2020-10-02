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
    
5.. We train the models with and without WCD converge to same-level training loss, then compare the optimal test loss. Thus may train several times to get a model converge to the similar train loss with another. 

         e.g., 

         /WCD/experimental_data/VGG16_corr_cifar10_nol2.out, VGG16_corr_cifar10_nol2_2.out, VGG16_corr_cifar10_nol2_3.out, VGG16_corr_cifar10_nol2_4.out  

         VS  

         /WCD/experimental_data/VGG16_normal_cifar10_nol2.out, VGG16_normal_cifar10_nol2_2.out, VGG16_normal_cifar10_nol2_3.out, VGG16_normal_cifar10_nol2_4.out, VGG16_normal_cifar10_nol2_5.out
         
         AND
         
         /WCD/experimental_data/VGG19_corr_cifar10_nol2.out, VGG19_corr_cifar10_nol2_2.out, VGG19_corr_cifar10_nol2_3.out, VGG19_corr_cifar10_nol2_4.out, VGG19_corr_cifar10_nol2_5.out
         
         VS
         
         /WCD/experimental_data/VGG19_normal_cifar10_nol2.out, VGG19_normal_cifar10_nol2_2.out, VGG19_normal_cifar10_nol2_3.out, VGG19_normal_cifar10_nol2_4.out, VGG19_normal_cifar10_nol2_5.out
         

All experimental data is saved in WCD/experimental_data     


Complexity_Measure:

         Results are shown in Complexity_Measure/complexity_measure_cifar10.ipynb and complexity_measure_cifar100.ipynb

         The raw models for complexity measure are saved in https://drive.google.com/drive/folders/1yfzvu-eQVntjTq_arabLBgGV2Ems98wQ?usp=sharing
