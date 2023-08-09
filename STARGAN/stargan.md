# [StarGAN : Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/pdf/1711.09020.pdf)     

------------------------------------------------------------------------------------------------------------------------------------       

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/0e6fc189-6043-4cda-a212-50d59d3f1b61'></p>     


------------------------------------------------------------------------------------------------------------------------------------       

## Motivation     

+ 최근 연구들은 image-to-image translation에서 큰 성공을 보여주었다.    

+ 하지만, 기존 접근법들은 **2개 이상의 domain을 처리하는 데 scalability와 robustness가 제한되어 있었다**.    

+ 왜냐하면, 각기 다른 모델들이 각 이미지 도메인 쌍마다 독립적으로 만들어졌기 때문이다.   

+ 이러한 한계점을 극복하기 위해 **StarGAN**이 등장했다.    

------------------------------------------------------------------------------------------------------------------------------------   

## Glossary     

+ **1. attribute** : 이미지에 있는 의미있는 특징들을 뜻하는데, 대표적으로 머리색, 나이, 성별 등이 있다.    

+ **2. attribute value** : attribute의 값을 의미함, 머리색의 경우 black, blond, brown 등이 있다.     

+ **3. domain** : 같은 attribute valueㄹ르 공유하는 이미지들의 집합을 말한다. 예를 들면 여성의 이미지들은 하나의 domain을 구성하고 남성의 이미지들은 또 다른 domain을 구성한다.     

------------------------------------------------------------------------------------------------------------------------------------   

### Introduction     

+ 맨 처음에 있는 사진은 CelebA dataset을 이용한 예시이다.    

+ 이러한 labeling dataset을 이용한 multi-domain image translation은 기존 모델에서는 비효율적이고 효과적이지 않았다.    
  + 이유 : k개의 domain 사이에서의 모든 mapping을 학습하기 위해서는 k(k-1)개의 generators가 학습되어야 하기 때문.    

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/e746e93b-a04e-4dde-abc2-de99b5df5e7e'width='500'></p>     

+ Figure 2 (a) 그림이 이를 나타낸 그림이다.   
  > **4개의 다른 domain**들 사이에서 이미지를 translation 시키기 위해서는 **4*(4-1) = 12개**의 네트워크가 필요하다.     
  > 또한, **각 데이터셋이 부분적으로 라벨링**되어 있기 때문에, **jointly training이 불가능**하다.

+ 이러한 문제들을 해결하는 모델이 바로 이 논문에서 제안하는 **StarGAN**이다.    
  > Figure 2 (b) 그림에서 볼 수 있듯이 StarGAN은 모든 가능한 domain들 사이의 mapping을 하나의 generator를 통해 학습한다.    


+ 고정된 translation, 예를 들면 흑발에서 금발로 바꾸는 translation을 학습시키는 것 대신, 이 single generator는 image와 domains 정보를 모두 input으로 넣고 유연하게 이미지를 알맞은 domain으로 바꾸는 것을 학습한다.     
  - 이러한 도메인 정보들을 표현하는 데에는 **binary or one-hot vector**와 같은 형식을 사용    

+ 학습하는 동안 **random하게 target domain label을 만들어내고** 모델이 유연하게 이미지를 target domain으로 변환하도록 학습시킨다.    

+ 더불어서, **mask vector**를 domain label에 추가함으로써, joint 학습이 가능하도록 하였다.    
  > 모델이 모르는 label을 **무시**할 수 있고 특정 데이터셋의 label들에 초점을 맞출 수 있다.    
  > 이러한 맥락에서, 모델은 얼굴 표정합성과 같은 task를 잘 수행할 수 있다.    
  > 아래에서 더 자세히 다룸    


------------------------------------------------------------------------------------------------------------------------------------       

## Method      

### 1. Muti-Domain Image-to-Image Translation       

+ Original GAN과 같은 맥락으로 목표는 여러 도메인간의 mapping을 학습하는 single generator를 학습시키는 것이다.    
  > 이를 위해서, input image $x$를 target domain label $c$의 조건에서 output image $y$로 변환시키도록 $G$를 학습한다.    
  > $$G\left(x,c\right) \rightarrow y$$    

+ random하게 target domain label $c$를 만들어내어 G가 유연하게 이미지를 변환시키도록 한다.    

+ 또한, auxiliary classifier를 통해 하나의 discriminator가 sources와 domain labels에 대해 **확률 분포**를 만들어내도록 한다.     
  > ![img1 daumcdn](https://github.com/WestChaeVI/GAN/assets/104747868/690fc2b3-a372-49dc-a2c7-c9400bc353a7)     

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/e4ec08a3-ae6e-4372-ad20-36bb8270f61e'width='500'></p>     


### 2. Loss function    

#### Adversarial Loss     
$$L_{adv} = {\mathbb{E}}\_{x} \left \[ \log{D_{src}\left(x\right)} \right \] + {\mathbb{E}}\_{x,c} \left \[ \log{\left(1 - D_{src}\left(G\left(x,c\right)\right)\right)} \right \]$$      

+ generator $G$가 image $G(x,c)$ 를 만들어내고 $D$는 진짜와 가짜 이미지들을 구분하는 역할을 한다.    

+ 여기서 $D_{src}$는 $D$에 의해 주어진 sources에 관한 확률 분포이다.     

+ $G$는 위의 loss 함수를 minimize하고자 하고 $D$는 maximize하고자 한다.    


#### Domain Classification Loss    

+ 주어진 input image $x$와 target domain label $c$에 대해 목표는 $x$를 target domain $c$로 분류된 output image $y$로 변환하는 것이다.    
  > 예를 들어, 여성 이미지 $x$를 타겟 도메인 라벨 $c$(남성)으로 변환하는 것이다.     

+ 이러한 조건을 만족시키기 위해 auxiliary classifier를 discriminator $D$ 위에 추가하고 domain classification을 수행한다.    

+ 즉, loss를 두 가지로 나누는데,    

  - 1) $D$를 maximize하기 위해 사용되는 **진짜 이미지**들에 대한 domain 분류 loss이다.     
      > $$L_{cls}^{r} = \mathbb{E}\_{x,c'} \left \[ -\log {D_{cls}\left(c'|x\right)} \right \]$$    
      >       
      > 이 함수를 최소화하기 위해서, $D$는 진짜 이미지 $x$를 original domain인 $c'$에 분류하는 것을 학습힌다. (즉, $D_{cls}\left(c'|x\right)$가 1에 가까워지도록 학습)         
  
  - 2) $G$를 minimize하기 위한 **가짜 이미지**들에 대한 domain 분류 loss이다.    
      > $$L_{cls}^{f} = \mathbb{E}\_{x,c} \left \[ -\log {D_{cls}\left(c|G\left(x,c\right)\right)} \right \]$$    
      >      
      > 즉, $G$를 최소화하기 위해 $D_{cls}\left(c|G\left(x,c\right)\right)$를 1에 가까워지도록 G를 학습한다.     


#### Reconstruction Loss    

+ **Adversarial loss**와 **classification loss**를 최소화하기 위해, $G$는 진짜같은, 올바른 target domain에 분류되는 이미지들을 만들어내도록 학습된다.    

+ 하지만, 위의 Loss를 최소화하는 것은 **변환된 이미지가 input image들의 내용을 보존한다는 것을 보장**하지 않는다.     

+ 이러한 문제를 완화하기 위해서, 논문에서는 generator에 **Cycle consistency Loss**를 적용하였다.    
  > $$L_{rec} = \mathbb{E}_{x,c,c'} \left\[ \lVert x - G \left( G \left(x,c\right), c'\right) \rVert _{1} \right\]$$    
  >       
  > Generator $G$는 변환된 이미지 $G\left(x,c\right)$와 original domain label $c'$를 input으로 하고, original image $x$를 다시 생성해내도록 시도한다. $\rightarrow$ **Reconstruction**    

+ 여기서 $L_1$ norm을 사용하였다. 하나의 generator를 두번 사용하는데     
  - 첫 번째는 original image input $x$를 target domain으로 변환시킬 때이고    
  - 두 번째는 변환된 이미지를 original image로 reconstruct할 때이다.    


#### Full objective     

+ 위에서 나온 loss들을 모두 정리해보면 다음과 같은 식을 도출한다.    

$$L_{D} = -L_{adv} + \lambda_{cls}L_{cls}^{r}$$      

$$L_{G} = L_{adv} + \lambda_{cls}L_{cls}^{r} + \lambda_{rec}L_{rec}$$     

$$where \lambda_{cls} and \lambda_{rec} are hyper-parameters$$     

$$We use \lambda_{cls} = 1 and \lambda_{rec} = 10 in all of our experiments$$     


------------------------------------------------------------------------------------------------------------------------------------       

## Training with Multiple Datasets    

+ StarGAN의 중요한 장점 중 하나는 다른 label들을 가지고 있는 여러 개의 dataset을 동시에 처리할 수 있다는 점이다.   

+ 그러나, 여러 개의 dataset으로부터 학습시킬 때, **각 데이터셋의 label 정보가 부분적으로만 알려져 있다는 것이 문제점이다**.   
  > 예를 들어, CelebA나 RaFD dataset에서 CelebA는 머리색과 성별과 같은 label을 포함한 반면, 행복, 분노와 같은 **표정**에 관련된 label은 가지고 있지 않다.    

+ 즉, **모든 dataset이 동등하게 label을 가지고 있는 것이 아니라 어떤 dataset은 특정 label만 가지고 있고 다른 dataset은 그 해당 dataset만의 label를 가지고 있는 것이다**.   

+ 이게 대체 어떤 식으로 문제가 작용하나?    
  > 변환된 이미지 $G\left(x,c\right)$에서 input image $x$를 Reconstructing 하는 과정에서 label vector $c'$에 대한 완전한 정보가 필요하기 때문이다.   
  >      
  > $L_{rec} = \mathbb{E}_{x,c,c'} \left\[ \lVert x - G \left( G \left(x,c\right), c'\right) \rVert _{1} \right\]$ 에서 label $c'$가 필요.     

### Mask vector     

+ 위의 문제점을 완화하기 위해, 논문에서는 **mask vector $m$**을 제안한다.    

+ 이 maxk vector는 StarGan이 **특정화되지 않은 label들은 무시**하고 **특정 dataset에서 존재하는 확실히 알려진 label에 focus를 맞추도록** 한다.     
  > 예를 들어, CelebA의 dataset에서 행복과 같은 label은 무시하고 머리색과 같은 label에 초점을 맞추도록 함.    

+ StarGAN에서는 mask vector $m$을 표현하기 위해 $n$차원의 one-hot vector를 사용하는데 $n$은 dataset의 수를 뜻한다.    

$$\tilde{c} = \left\[c_1, ..., c_n, m \right\]$$    
  > list 형태가 아닌 concatenation을 뜻함. matrix 형태    

+ $c_i$는 $i$ 번째 데이터셋의 label에 대한 vector를 뜻한다.    
  - 알려져 있는 label의 vector $c_i$는 binary attributes에 대해서는 binary vector로 표현될 수 있고,     
  - categorical attributes에 대해서는 one-hot vector로 표현될 수 있다.     
  - 남은 n-1개의 알려지지 않은 라벨에 대해서는 zero값으로 지정한다.    

+ 이 논문의 실험에서는 CelebA와 RaFD dataset을 사용하였으므로 $n = 2$가 된다.    


### Training Strategy     

+ Training 과정에서는 , domain label $\tilde{c} = \left\[c_1, ..., c_n, m \right\]$ 를 generator의 input으로 사용하였다.     
  > 다시 한번 강조하지만, 이를 통해 **generator가 알려지지 않은 label에 대해서는 무시를 하게 된다 ($\because$ zero vector)**.    
  >      
  > 또한 확실하게 주어진 label에 focus를 맞춰 학습하게 된다. generator의 구조는 **input label의 차원을 제외하고는** 하나의 데이터셋에 대해 학습할 때와 같은 구조이다.   

+ 반면, 이 논문에서는 discriminator의 auxiliary classifier를 확률 분포를 만들어내기 위해 확장시켰다.    

+ 그 후 다양한 task에 대한 학습을 하였고 여기서 **discriminator는 classification error만을 minimize하게 된다**.   
  > 예를 들어, CelebA dataset 이미지에 대해 학습할 때에는 discriminator가 CelebA attributes(ex. hair color, gender)에 대해서만 classification error를 minimize하게 되고 RaFD의 표정과 같은 특징들은 무시한다.    

+ 이러한 setting에서, 다양한 dataset을 번갈아가며 학습하며 discriminator는 모든 dataset에 관한 discriminative 특징들을 학습하고 generator는 모든 label들을 제어하는 것에 대해 학습하게 된다.    

------------------------------------------------------------------------------------------------------------------------------------     

## Implementation   
    

### Improved GAN Training     

+ 학습 과정을 안정화시키고 더 좋은 quality의 image들을 만들어내기 위해 이 논문에서는 $L_{adv}$ 수식을 **WGAN-GP**로 대체하였다.    

$$L_{adv} = {\mathbb{E}}\_{x} \left \[ D_{src}\left(x\right) \right \] - {\mathbb{E}}\_{x,c} \left \[ \left( D_{src}\left(G\left(x,c\right)\right)\right) \right \] - \lambda_{gp}\mathbb{E}_{\hat{x}} \left \[ \left(\lVert \triangledown\_{\tilde{x}} D\_{src}\left(\tilde{x}\right) \rVert\_{2} - 1 \right)^2 \right \]$$    
> where $\tilde{x}$ is sampled uniformly along a straight line between a pair of a real and a generated images. We use $\lambda_{gp} = 10$ for all experiments.    


### Network Architecture     

#### Generator    

+ CycleGAN 구조를 약간 차용한 StarGAN을 사용한다.   
  - downsampling을 위한 2-stride convolution layers     
  - 6 residual blocks    
  - upsampling을 위한 2-stride transposed convolution    

<table>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/73d5caf8-9549-4be4-bd72-4f36a1627ea4'></p>
  </td>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/dbcaa650-bbb7-437d-a761-dc7c16c401ee'></p>
  </td>
</table>     

#### Discriminator    

+ PatchGAN을 사용    
+ $n_d$는 domain의 개수     

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/f78c2d5c-1ef0-4e48-aef1-e31a49166b0e'></p>

+ 또한 generator에 instance normalization을 사용하고 discriminator에는 사용하지 않는다.      

------------------------------------------------------------------------------------------------------------------------------------       

## Experiments    


## 개인적으로 느낀 점    

