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

+ 또 흥미로운 점으로는 **random noize z를 일반적으로 사용하지 않는다**. 이는 Generator가 이를 무시하도록 학습되기 때문이라고 한다.     
  
