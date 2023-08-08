# [Image-to-Image Translation with Conditional Adversarial Network](https://arxiv.org/pdf/1611.07004.pdf)     

## Motivation     

+ Computer Vision에는 방대한 task들이 존재하는데, 많은 문제들을 input image를 어떠한 다른 양상의 output image로 **"Translating"** 하는 것이라 볼 수 있다.     

### 1. Problems of Traditional Methods     

+ 이러한 여러 Computre Vision에 존재하는 여러 task들을 각각의 task하고 생각하고 해당 task들에 맞추어 **model structure** 혹은 **Loos** 를 Customize을 수행    

  + 논문 저자 曰 "굳이 task들을 나눠서 접근해야 하나??"    

  + Why? 결국 task들이 하고자 하는 것은 **pixel들로 부터 pixel들을 예측하는 문제**로 보고 **공통적인 목표***를 가진다고 생각했기 때문    

+ 결국, **모든 task들을 같은 문제로 바라보고 공통으로 사용할 수 있는 framework**를 제안     

### 2. Problems of image-to-image translation with CNN     

+ Image-to-image translation task들에 대해서 CNN을 사용한 방법들이 많이 연구되었는데     
  **Loss를 effective하게 잘 design하지 않으면 결과가 제대로 나오지 않는 문제**가 발생    

  + ex) CNN에게 pred, GT 간의 **$L_2$ distance를 학습하게 만들면 outputs을 평균화하는 방향으로 minimize 되기 때문**에 output 이미지가 **blurry** 해진다.    

+ 위 두 가지 문제점에 대한 대책으로 GAN을 제안한다.  GAN은 단순히 이미미 간의 mapping을 수행하는 것 뿐만 아니라 두 확률 분포를 비슷하게 만들도록 학습되기 때문에 blurry함을 방지할 수 있다.    
## Method      

### Objective function    

+ 논문에서 CGAN을 base로 한다.   

$$L_{CGAN}\left(G,D\right) = \mathbb{E}_y \left \[ \log{D\left(x,y\right)} \right \] + \mathbb{E} _{x,z} \left \[ \log{\left(1 - D\left(G\left(x,z\right)\right)\right)} \right \]$$    

+ 원래 GAN은 생성하는 Output이 어떤 종류가 나올 지 제어하는 것이 불가능 했었는데 위 식에서 볼 수 있듯이 원하는 Class의 정보를 D와 G의 input과 같이 조건으로 넣어 줌으로써 제어가 가능하게 만든 것이다.    

+ 논문에서는 이전 연구에서 traditional loss와 GAN loss를 같이 사용하는 것이 효과적이었기 때문에 **L1 loss를 같이 사용**한다.    
  > 즉, Generator가 Discriminator를 속이는 것 뿐만 아니라 GT 이미지와 비슷해지도록 학습이 될 것 이다.    
  > **L2보다 L1이 blurry output이 만들어질 확률이 낮기 때문**    

+ 또 흥미로운 점으로는 **random noize** $z$ **를 일반적으로 사용하지 않는다**. 이는 Generator가 이를 무시하도록 학습되기 때문이라고 한다.     
  > 즉, 넣어도 별 소용이 없거나 안 좋은 영향을 줄 수 있다는 것이다.   

+ 이 $z$ 가 없기 때문에 mapping 과정에서 **stochastic** 한 결과를 내지 못하고 deterministic 한 결과만 내보내는 문제가 있지만, **저자는 다양한 결과를 내기 보다는 있을 법한 결과를 내는 것에 더 의의가 있다고 말하면서 feature work로 남김**   

+ **대신 Stochastic 성질이 생기도록 layer에 dropout을 적용**    
  > Dropout이 기존과 다르게 test에서도 남아 있는 채로 prediction을 한다.

$$L_{pix2pix}\left(G,D\right) = \mathbb{E}\_y \left \[ \log{D\left(x,y\right)} \right \] + \mathbb{E}\_{x} \left \[ \log{\left(1 - D\left(G\left(x\right)\right)\right)} \right \] + \lambda L_{L1}\left( G \right)$$   

### Network Architecture    

#### Skip Connection     
<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/d798b4e0-8cce-455c-a7e3-98cbaf7aa86e'></p>     

+ image-to-image translation task들을 살펴보면 model structure은 그대로 유지하면서 surface apperance만 바꾼다.     
  > 즉, **Input의 structure가 거의 그대로 output에 사용**된다는 것이다.   

+ 이를 기반으로 논문에서는 Encoder-Decoder network에 **residual connection**을 사용하여 input의 information을 공유하고 단순 concatenation을 통해서 이를 decoder의 features와 aggregate한다.      


#### PatchGAN    

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/abccd19e-732d-4a85-8b05-51bb6b86a913' width='600' height='400'></p>    

+ $L_2$ 와 $L_1$ 은 high-frequency 정보는 잘 살려내지 못해서 blurry 하게 나오지만 low-frequency 정보는 잘 잡아낸다.    
  > high/low-frequency가 무엇인가?    
  >     
  > <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/2301de88-1b3f-4389-a7d8-be2f07866c5e'></p>     
  >      
  > + high-frequency information : 주변영역과 색의 차이가 크게 나는 부분, 객체의 가장자리 부분 (경계면)       
  > + low-frequency information : 주변영역과 색의 차이가 적은 부분, 객체의 중심 부분     
  >       
  > 다시 말해, $L_2$ 와 $L_1$ 은 객체 중심 부분의 정보만 잡아낸다는 의미이다.    

+ 따라서 저자들은 GAN discriminator는 high-frequency structure을 모델링하도록 제한해두고 $L_1$ term이 low-frequency 정보를 잡아내도록 역할 분담 시켰다.     

+ high-frequency 정보를 잘 만들어내도록 할 때, 이미지 전체를 보는 것보다 local하게 보는 것이 더 효율적이다. (PatchGAN의 motivation)     
  + Patch GAN 간단 설명 : Disciminator가 진짜 이미지와 가짜 이미지를 구별할 때 **전체 이미즐ㄹ 보는 것이 아니라 Patch(NxN) 단위로 prediction**하는 것이다.   
  + 이때 patch size N은 전체 이미지 크기보다 훨씬 작기 때문에 **더 작은 parameters와 빠른 running time**을 가진다.    

+ Pix2Pix에서는 256x256 크기의 GT와 generated fake image를 concat한 후에 위 그림처럼 convolution layer를 거쳐 최종 feature map이 30x30x1 크기를 가지게 돤다.     
  > output feature nmp의 픽셀 1개의 receptive field는 입력 이미지의 70x70에 해당한다.    

#### Optimization and Inference    

+ train과 test 과정에서 몇 가지 특이점을 가지고 있다.    
  1. 먼저 Generator G를 학습할 때 $\log{\left(1 - D\left( x, G\left(x,z\right)\right)\right)}$ 를 minimize하기 보다는 $\log{\left( D\left( x, G\left(x,z\right)\right)\right)}$ 를 maximize하도록 바꿨다.     
    > 그 이유는 Generator가 학습할 때 초반에 안좋은 성능을 빨리 벗어나기 위함이다.     
    >      
    > GAN loss를 이용한다면 거의 다 이와 같은 방식으로 한다.      
    >      
    > 자세한 내용은 [Generator Loss 부분](https://github.com/WestChaeVI/GAN/blob/main/SRGAN/srgan.md)      

  2. Discriminator를 학습할 때 loss function을 2로 나누어서 Generator보다 더 천천히 학습되도록 만들었다.    
    > 이는 학습 초기에 G가 가짜 이미지를 형편없게 만들다 보니 상대적으로 D의 task가 쉬워 학습이 진행되지 않는 문제를 해결하기 위함이다. ~~사실 기존 GAN에서도 같은 방법을 사용해 특이하진 않다.~~     

  3. Test 시 Dropout을 그대로 사용한다는 점과 Batch normalization 또한 train batch의 $\mu$와 $\sigma$를 사용하지 않고 test batch의 $\mu$와 $\sigma$를 사용한다는 점이다.    

## Experiments    

### Analysis of the objective function     

<table align='center'>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/ae508ff7-4c14-45a5-9b05-cb9565c7411c' height='400'></p>   
  </td>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/a81177bf-bbaf-44b0-abbb-a6f593578ffa'></p>    
  </td>
</table>

+ $L_1$ loss와 cGAN loss를 같이 썼을 때 가장 높은 성능을 보이며 qualitative 측면에서도 좋은 것을 확인할 수 있다.    

+ cGAN loss만 사용할 때의 이미지가 상대적으로 sharp한 결과를 보여주지만 **visual artifacts가 생기는 문제**가 있다.       


### Analysis of the generator architecture    

<table align='center'>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/5ae17dce-1dc0-4baf-bf8e-1dd3266ef18b'></p>   
  </td>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/04f291da-1fd0-4e6c-8232-5460f85c18f3' width='500' height='200'></p>      
  </td>
</table>     

### Analysis of the PatchGAN architecture    

<table align='center'>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/5ae17dce-1dc0-4baf-bf8e-1dd3266ef18b'></p>   
  </td>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/04f291da-1fd0-4e6c-8232-5460f85c18f3' width='500' height='200'></p>      
  </td>
</table>     

+ PatchGAn의 patch size를 변경해가면서 성능을 살펴본 실험이다.    
  - 1x1의 경우 매우 blurry한 것을 볼 수 있고      
  - 16x16은 충분히 sharp하지만 artifacts가 다수 생겨났고     
  - 70x70일 때 그러한 artifacts가 사라진 것을 볼 수 있다.     
  - 286x286은 quality 측면에서 큰 변화가 없고 오히려 FCN-score가 낮은 결과가 나왔다.   

+ 이 실험결과에 대해서 저자들은 full image GAN이 70x70 PatchGAN보다 **훨씬 많은 parameter 수와 더 큰 depth를 가지고 있기 때문에 학습이 더 어려워** 나타난 결과라 추측한다.    


### Other analyzes   

<table align='center'>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/ea700676-60f1-4954-9606-f63c7a541a39'width='700'></p>   
  </td>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/054ee7e5-fa3d-45eb-92d0-4fed487bd76d'></p>      
  </td>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/77a716a5-e4e7-412d-8f06-e5ab48410e6b'></p>      
  </td>
</table>      


+ 이 외에도 다양환 task들을 실험하였는데, 주목할만한 점은 **photo로부터 Segmentation labels를 생성하는 task에서는 단순 $L_1$ Regression이 cGAN보다 더 좋은 성능**을 보여준다는 것이다.      

+ 저자들은 처음으로 GAN을 사용하여 성공적으로 labels을 generating 했다는 부분을 언급하고 cGAN objective function 관점에서 $L_1$ loss 보다 더 ambiguous 하여 상대적으로 discrete한 labels를 생성하는 능력이 떨어졌다라고 설명한다.     



## 개인적으로 느낀 점    

### Test Dropout     

+ Dropout를 Test 시에도 활성화된 상태로 적용한다고 하였다.     

+ Generator는 학습한 뒤에도 언제든 새로운 input에 대한 sample를 생성할 수 있어야 한다.    

+ 논문 초반에 저자가 강조했듯이 다양한 결과보다는 있을 법한 결과를 도출하는 것에 의의를 두었기 때문에     

+ 따라서, Dropout을 test시에도 활성화해 놓으면 Generator가 새로운 입력에 대해 더 일반화된 결과를 생성할 수 있기 때문이지 않을까라는 개인적인 생각이다.     


### Test Batch Normalization     

+ Test 시에 test batch의 평균과 분산을 이용해서 진행한다고 하였다.     

+ Batch Normalization이 모델의 성능을 향상시키고 일반화를 돕는데,      

+ 학습된 모델이 test 시에도 일관된 예측을 내놓기 위해서는 해당 batch의 평균과 분산을 사용하는 것이 맞다고 생각한다.    

+ 즉, test 시에 전체 학습 데이터셋의 평균, 분산을 사용하지 않는 이유는 모델의 일반화 능력을 유지하는 데에 있어서 현재 입력에 대한 통계정보가 더 중요하다 생각하지 않았을까 생긱이 든다.      

