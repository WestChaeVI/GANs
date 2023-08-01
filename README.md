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

+ Dataset : [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
+ crop_size = 96
+ upscale_factor : 4
+ Batch_size : 64 (validset : 1)    
+ epochs : 250


  
------------------------------------------------------------------------------------------------------------       

## [Pix2pix(2018)](https://github.com/WestChaeVI/GAN/blob/main/PIX2PIX/pix2pix.md)    

+ Dataset : [facades](https://www.kaggle.com/datasets/balraj98/facades-dataset)
+ Batch_size : 32
+ epochs : 100 
+ lambda_pixel : 100 (Loss_func_pix weights)
+ patch : (1, 256//2**4, 256//2**4)
+ lr : 2e-4
+ beta1 = 0.5
+ beta2 = 0.999      
<p align="center">
<img src='https://github.com/WestChaeVI/CNN-models/assets/104747868/59fb009b-8140-419f-8a86-085aff830f6f' width='600' height="600">
</p>    

------------------------------------------------------------------------------------------------------------       

## [StarGAN(2018)](https://github.com/WestChaeVI/GAN/blob/main/STARGAN/stargan.md)    

+ Dataset : [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
+ Batch_size : 64    
+ nz : 100 (Size of z latent vector)
+ iters : 200000
+ lr : 0.0001
+ beta1 : 0.5
+ beta2 : 0.999    

<table style="margin-left: auto; margin-right: auto;">
  <th>
    <p align='center'>Input</p>
  </th>
  <th>
    <p align='center'>Young</p>
  </th>
  <th>
    <p align='center'>Blond hair, Yonng</p>
  </th>
  <th>
    <p align='center'>Black hair, Male, Young</p>
  </th>
  <tr>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/CNN-models/assets/104747868/41378ae8-7c61-4110-9751-d8949f292cee' width='500'>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/9b3447bb-015d-45a9-a69b-efcb40acbb47' width='500'>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/ff59f238-3ba8-45be-a33a-d0866e760aaa' width='500'>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/CNN-models/assets/104747868/e3263b8f-998e-4ed6-9d1f-67d4742bcecc' width='500'>
      <p>
    </td>
  </tr>
</table>  

------------------------------------------------------------------------------------------------------------       

## [CycleGAN(2020)](https://github.com/WestChaeVI/GAN/blob/main/CycleGAN/cyclegan.md)    

+ Dataset : [horse2zebra](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset)  
+ Batch_size : 1    
+ epochs : 200
+ lr : 0.0002
+ beta1 : 0.5
+ beta2 : 0.999    
+ deacy_after = 100    
+ lambda_cyc : 10.0
+ lambda_idt : 0.5

### Test Result (Orginal - translated - reconstructed)     

+ Horse to Zebra      

![image](https://github.com/WestChaeVI/GAN/assets/104747868/97a32236-a0c0-4ea8-b6b7-80bd8193c56f)       

+ Zebra to Horse   

![image](https://github.com/WestChaeVI/GAN/assets/104747868/3dfc408f-e777-40b7-ada0-06c6e6917f0b)    


------------------------------------------------------------------------------------------------------------       
