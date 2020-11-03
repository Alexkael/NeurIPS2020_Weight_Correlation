**How does Weight Correlation Affect theGeneralisation Ability of Deep Neural Networks?** [arxiv](https://arxiv.org/abs/2010.05983)

**Gaojie Jin, Xinping Yi, Liang Zhang, Lijun Zhang, Sven Schewe, Xiaowei Huang**

**Paper Abstract**
This paper studies the novel concept of weight correlation in deep neural networks and discusses its impact on the networks' generalisation ability. For fully-connected layers, the weight correlation is defined as the average cosine similarity between weight vectors of neurons, and for convolutional layers, the weight correlation is defined as the cosine similarity between filter matrices. Theoretically, we show that, weight correlation can, and should, be incorporated into the PAC Bayesian framework for the generalisation of neural networks, and the resulting generalisation bound is monotonic with respect to the weight correlation. We formulate a new complexity measure,  which lifts the PAC Bayes measure with weight correlation, and experimentally confirm that it is able to rank the generalisation errors of a set of networks more precisely than existing measures. More importantly, we develop a new regulariser for training, and provide extensive experiments that show that the generalisation error can be greatly reduced with our novel approach.


version: 
         
         python 3.5+

         tensorflow 1.13.1-gpu
         
         keras 2.2.4
         
WCD method experiment:

Results are shown in /WCD/experimental_data/AAA_optimil_loss.ipynb

1..  run: 

         FCN.py                 fully connected network without WCD

         FCN_corr.py            fully connected network with WCD
         
         VGG.py                 VGG without WCD
         
         VGG_corr.py            VGG with WCD, !!!Uncomment sgwn_CNN.py line 87 or 89 or 91

2..  

         sgwn_CNN.py: wcd method for CNN

         sgwn.py: wcd method for FCN
    
3.. We train the models with and without WCD **--converge to same-level train loss (training repeatedly, the model with invariant setting may converge to different train loss)--**, then compare the optimal test loss. Thus, we may train the same model (WCD or without WCD) several times to get a model converge to the similar train loss with another one (without WCD or WCD). **--e.g.,--**

         /WCD/experimental_data/                  train loss                                          train loss  
         
         comparable group:
         VGG11_corr_cifar10_nol2.out  (converge to)  1.145    VS      VGG11_normal_cifar10_nol2.out    1.140
         
         unused:
         VGG11_corr_cifar10_nol2_2.out               1.123            VGG11_normal_cifar10_nol2_2.out  1.161
         VGG11_corr_cifar10_nol2_3.out               1.342            VGG11_normal_cifar10_nol2_3.out  1.172
         VGG11_corr_cifar10_nol2_4.out               1.231            VGG11_normal_cifar10_nol2_4.out  1.199
         VGG11_corr_cifar10_nol2_5.out               1.121            VGG11_normal_cifar10_nol2_5.out  1.285
                
         AND
         
         comparable group:
         VGG16_corr_cifar10_nol2.out  (converge to)  1.008    VS      VGG16_normal_cifar10_nol2.out    1.015
         
         unused:
         VGG16_corr_cifar10_nol2_2.out               0.995            VGG16_normal_cifar10_nol2_2.out  1.086
         VGG16_corr_cifar10_nol2_3.out               1.098            VGG16_normal_cifar10_nol2_3.out  0.980
         VGG16_corr_cifar10_nol2_4.out               1.045            VGG16_normal_cifar10_nol2_4.out  1.004
         VGG16_corr_cifar10_nol2_4.out               1.022            VGG16_normal_cifar10_nol2_5.out  0.993
                                                                      
         AND
         
         comparable group:
         VGG19_corr_cifar10_nol2.out                 1.021    VS      VGG19_normal_cifar10_nol2.out    1.022                
         
         unused:
         VGG19_corr_cifar10_nol2_2.out               0.987            VGG19_normal_cifar10_nol2_2.out  0.984
         VGG19_corr_cifar10_nol2_3.out               1.014            VGG19_normal_cifar10_nol2_3.out  1.028
         VGG19_corr_cifar10_nol2_4.out               1.010            VGG19_normal_cifar10_nol2_4.out  0.988
         VGG19_corr_cifar10_nol2_5.out               1.056            VGG19_normal_cifar10_nol2_5.out  1.033
         
         All experimental data is saved in WCD/experimental_data     


5.. Additional experimental data is saved in WCD/additional. However, as the networks are small and datasets are more complicated, the errors are pretty high and the results are more random.

         For cifar100, last three layers of VGG adjusted to 120, 120, 100
         
         For cal256, last three layers of VGG adjusted to 257, 257, 257

Complexity_Measure experiment:

         Results are shown in Complexity_Measure/complexity_measure_cifar10.ipynb and complexity_measure_cifar100.ipynb

         The raw models for complexity measure are saved in https://drive.google.com/drive/folders/1yfzvu-eQVntjTq_arabLBgGV2Ems98wQ?usp=sharing

**Citation**
If you use our code in your research, please cite
@article{jin2020does,
  title={How does Weight Correlation Affect the Generalisation Ability of Deep Neural Networks},
  author={Jin, Gaojie and Yi, Xinping and Zhang, Liang and Zhang, Lijun and Schewe, Sven and Huang, Xiaowei},
  journal={arXiv preprint arXiv:2010.05983},
  year={2020}
}
