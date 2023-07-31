# GAN
Generative Adversarial Network paper summary and implementation

------------------------------------------------------------------------------------------------------------    

## [DCGAN(2016)](https://github.com/WestChaeVI/GAN/blob/main/DCGAN/dcgan.md)     

+ Dataset : [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
+ Batch_size : 128    
+ nz : 100 (Size of z latent vector)
+ epochs : 20
+ lr : 0.0002
+ beta1 : 0.5    
<p align="center">
<img src="https://github.com/WestChaeVI/CNN-models/assets/104747868/61d00cea-c8b2-4155-8d03-114b017cc031" width="850" height="400">  
</p>     

------------------------------------------------------------------------------------------------------------       

## [WGAN(2017)](https://github.com/WestChaeVI/GAN/blob/main/WGAN/wgan.md)    

+ Dataset : [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
+ Batch_size : 64    
+ nz : 100 (Size of z latent vector)
+ **n_critics : 5**
+ epochs : 20
+ lr : 0.0001
+ beta1 : 0.5
+ beta2 : 0.9
+ **weight_cliping_limit : 0.01**
+ **lambda_gp : 10(gradient penalty**    

<table style="margin-left: auto; margin-right: auto;">
  <th>
    <p align='center'>Visualization</p>
  </th>
  <th>
    <p align='center'>Interpolation</p>
  </th>
  <tr>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/CNN-models/assets/104747868/f2e423e4-2d7a-4cda-ae20-05e0068e93e3' width='500'>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/CNN-models/assets/104747868/c2e0ef8d-7125-430a-bff2-002d34ff006d' width='700' height="300">
      <p>
    </td>
  </tr>
</table>    

------------------------------------------------------------------------------------------------------------       

## [SRGAN(2017)](https://github.com/WestChaeVI/GAN/blob/main/SRGAN/srgan.md)    


------------------------------------------------------------------------------------------------------------       

## [Pix2pix(2018)](https://github.com/WestChaeVI/GAN/blob/main/PIX2PIX/pix2pix.md)    


------------------------------------------------------------------------------------------------------------       

## [StarGAN(2018)](https://github.com/WestChaeVI/GAN/blob/main/STARGAN/stargan.md)    


------------------------------------------------------------------------------------------------------------       

## [CycleGAN(2020)](https://github.com/WestChaeVI/GAN/blob/main/CycleGAN/cyclegan.md)    


------------------------------------------------------------------------------------------------------------       
