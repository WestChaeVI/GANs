# [Unspurvised Representation Learning With Deep Convolutional Generative Adversarial Networks(ICRL 2016)](https://arxiv.org/pdf/1511.06434.pdf)    

------------------------------------------------------------------------------------------------   

## Motivation    
+ 가존 GAN의 경우 **불안정**하게 학습을하고, generator가 종종 **무의미한** output을 출력하는 문제점이 있었음.
+ 그래서 왜 이러한 문제점이 발생하는지에 대해 **GAN이 무엇을 학습하는지, 또 학습과정 중의 중간 representation을 직접 시각화 함**.

------------------------------------------------------------------------------------------------    

## Contribution    
저자는 논문의 contribution으로 다음의 네 가지를 제시함.    

+ Stable training process: 대부분의 설정에서 GAN의 학습 안정성을 높인 Convolutional GANs 구조를 제안 (**DCGANs**)     
+ Trained Discriminator: **Image Classification tasks**에서 학습된 D를 사용 → 이 경우 다른 비지도학습 기반 알고리즘보다 더 높은 성능을 보여줌.     
+ Visualizing GANs: GAN의 convolution 필터를 visualize & **특정 필터가 특정 물체를 그리도록 학습됨**을 경험적으로 보여줌.     
+ G's vector arithmetic properties: 생성된 샘플의 semantic quality를 쉽게 **조작**할 수 있도록 하는 **vector arithmetic properties**를 발견함.     

------------------------------------------------------------------------------------------------ 

## Model Architecture    
<p align='center'>
<img src="https://github.com/WestChaeVI/GAN/assets/104747868/0f29696d-f15a-45e5-a1bb-9bd85591dac1">
</p>     

기존 지도학습에서 일반적으로 사용되는 CNN 아키텍처를 사용하여 GANs을 scale 하려는 시도를 했었지만, 다소 어려움이 있었음.     
그래서 다음과 같은 기법들로 모델 구조를 만듦.     

+ 모든 Pooling Layer를 Convolution Layer로 교체
  (G : Fractional-Strided Conv / D : Strided Conv)     
  > Fractional-Strided Conv     
    : Stride 값을 1보다 작은 분수로 두어, 결과적으로 출력의 크기가 커짐.     
  > Strided Conv     
    : Stride 값을 1보다 크게 잡아, 결과적으로 Pooling Layer처럼 출력의 크기가 작아짐.      
  >     
  > G에 이 approach를 사용하면 network가 각각의 spatial downsampling/upsampling을 학습할 수 있고, D 또한 학습할 수 있게 됨.    

+ Generator, Discriminator 둘 다 Batch Normalization 적용    
  > G의 경우 mode collapse 방지.     
  >      
  > 모든 Layer에 BatchNorm을 다 적용하면 sample oscillation과 model instability가 발생.     
  > 이를 해결하기 위해 G의 출력 Layer와 D의 입력 Layer에 BatchNorm을 적용하지 않았음.     

+ 더 깊게 쌓기 위해서 모든 FC layer 삭제
  > FC layer는 전체 CNN보다 더 많은 parameter를 가지고, feature의 위치 정보들도 사라지는 단점들이 있음.     

+ Activation function    
  > G : 출력을 제외한 모든 Layer에 ReLU, 출력은 Tanh 사용     
  > D : 모든 Layer에 Leaky ReLU 사용.     
  >     
  > 논문 저자는 bounded activation를 사용함으로써 모델이 training distribution의 color space를 saturate, cover 하는 것을 더 빨리 학습하게 해준다는 것을 관찰했다고 함.      

------------------------------------------------------------------------------------------------    

## Details of Adversarial Training    

- No Pre-processing besides scaling to the range of tanh activation $[-1, 1]$
- SGD with batch size of 128
- weight init: zero-centered normal with s.t.d 0.02
- Leaky ReLU: slope = 0.02 with leak
- Adam Optimizer with learning rate = .0002
  - momentum $\beta_1$ = 0.5 ($\because$ training oscillation and instability with $\beta_1$ of 0.9)    

------------------------------------------------------------------------------------------------   

## LSUN    

+ Generator model들의 output의 visual quality가 개선되면서, over-fitting과 training samples의 memorization 문제가 대두됨.
+ DCGAN이 데이터의 양과 sample의 resolution이 증가(향상)하는 상황에도 잘 대처할 수 있음을 보여주기 위해 300만장의 LSUN 데이터셋을 활용함. (data augmentation 적용X)
  > small learning rate와 SGD를 써서 보여줌.    

------------------------------------------------------------------------------------------------     

## Deduplication    

+ G가 단순히 training example을 기억해서 모방하는 현상을 방지하기 위해, 간단한 이미지 중복제거 단계를 거침.    
+ 중복 제거를 위해 DAE (Denoising Autoencoder with dropout regularization + ReLU activation)를 fitting시킴.     
  > **3072-128-3072** (입출력 차원 3072, latent 차원: 128)      
     
+ 결과적으로 latent code 레이어를 ReLU activation을 통해서 binarize시킴으로써 semantic hasing을 수행     
  (이를 통해 중복 제거를 linear time, 즉 $O(n)$에 수행. 아마 해싱의 최악 시간복잡도가 $O(K)$임에서 비롯되지 않았을까 싶음)     
 → 간단히 해시 충돌에 대해 Visual inspection을 수행한 결과, 추정된 FPR(False Positive Rate)이 **0.01 미만**으로 높은 정확도를 보여주었음.     
+ 실제로 적용해본 결과 대략 275,000개의 중복을 제거하면서 비교적 높은 recall 값을 가질 것을 시사함.      

------------------------------------------------------------------------------------------------    

## Investigating And Visualizaing The Internals Of The Networks     

### Walking In The Latent Space    

+ Model이 학습한 Manifold를 관찰하고 이해해봄.     
+ Signs of Memorization(급격한 전환이 있는 경우)와 공간이 계층적으로 붕괴되는 방식에 대해 알 수 있음.    
+ 이 Latent Space를 확인해보는 것이 이미지 생성에 의미론적 변화(개체 추가 및 제거)를 초래하는 경우 모델이 관련성 있고 흥미로운 표현을 학습했다고 추론할 수 있음.    
  > 아래 이미지의 맨 아래열을 보면 TV가 창문으로 바뀌는 것을 볼 수 있음.    

<p align='center'>
<img src="https://github.com/WestChaeVI/GAN/assets/104747868/22f073db-6d7d-4623-aefb-6b72f4de8561" width=700 height=700>
</p>     


### Visualizing The Discriminator Features    

+ 각각의 이미지들이 어떤 부분을 학습했는지 보여주고, Black box를 조금이나마 풀려고 했음.  
+ guided backpropagation을 통해 D의 feature map을 시각화한 결과, 실제로 침대나 창문 등 전형적인 침실의 특징 부분이 활성화된 것을 알 수 있었음.

<p align='center'>
<img src="https://github.com/WestChaeVI/GAN/assets/104747868/be091999-3406-4f10-b62d-da6c08cc495b">
</p>     

------------------------------------------------------------------------------------------------       

## Manipulating The Generator Representation  

### Forgetting To Draw Certain Objects 

+ G가 어떤 표현을 학습했는지에 대해 알아보려고 함. generator로부터 이미지 내에 있는 **window**를 없애 보는 실험을 수행.     
+ 150개의 샘플을 대상으로 52개의 window bounding box를 직접 그림.      
+ 이후, 뒤에서 2번째의 convolution layer에서 logistic regression을 수행해서 feature activation이 window에 있는지 아닌지를 판별.    
+ window에 activate된 feature는 모두 삭제함.      

<p align='center'>
<img src="https://github.com/WestChaeVI/GAN/assets/104747868/7f25b652-ffca-4f0e-b737-93e86a43241b">
</p>     



### Vector Arithmetic on face samples    

+ word embedding 시 vector 간 연산으로 representation이 잘 되었는지를 확인하는 방법을 face generation에도 그대로 적용.      

+ 하지만 하나의 sample 단위로 수행하면 결과가 좋지 않아서 각 컨셉 (여성/남성, 안경/안경 없음 등) 별로 Z를 **평균**하여 벡터 연산을 수행한 결과 안정적인 생성 결과가 나올뿐 아니라 실제로 연산의 결과와 부합하는 생성 결과가 나왔다.     
> vector("King") - vector("Man") + vector("Woman") = vector("Queen")    

+ 이러한 벡터연산을 통해 unsupervised model에서도 conditionally generation이 가능하다는 것을 동시에 보여주고 있음.     

<p align='center'>
<img src="https://github.com/WestChaeVI/GAN/assets/104747868/ff2f5db3-6e8a-4cf1-ad24-d39c0a32e2c1">
</p>     

------------------------------------------------------------------------------------------------     

## Conclusion & Future Work(limitation)      

+ GAN 훈련을 위한 보다 Stable architecture를 제안함.   
+ 적대적 네트워크가 지도 학습 및 generative modeling을 위해 이미지의 좋은 표현을 학습한다는 증거를 제공함.   

+ 여전히 몇 가지의 모델 불안정성이 남아 있음. 모델이 학습될 수록 가끔씩 convolution filter 중 일부분이 값이 진동(oscillation)하는 현상이 발생함.    
+ 다른 분야로의 확장 such as video and audio.   

------------------------------------------------------------------------------------------------      

### 개인적으로 느낀 점   
     
     
생성 분야의 논문들은 대부분 모델도 모델이지만 Loss function을 어떻게 정의해서 성능을 높일 수 있을까하는 방식으로 가지만,    
DCGAN은 이와는 다르게 단지 기존 GAN의 불안정한 학습을 해결하고자 기존 GAN 구조와 CNN architecture를 이용한 모델을 만듬.   

사실 모델 구조까지만 읽었을 때는 그렇게 까지 엄청 대단하다고 느끼진 않았다..  

그러나, 논문 뒤에 나오는 여러 실험들과 더불어 학습된 Manifold를 그려보고 분석한 점, G와 D가 무엇을 학습하는지 보려고 한 것,   
특정 target('window')를 잡고 지우려고 해본 것, 그리고 벡터 연산을 통해 unsupervised model에서도 conditionally generation이 가능하다는 것 등등   
정말 깊은 생각과 관점으로 많은 실험들을 통해 독자들로 하여금 다양한 인사이트를 제공할 것으로 생각된다.

