# [Wasserstein GAN (2017)](https://arxiv.org/pdf/1701.07875.pdf)    

------------------------------------------------------------------------------------------------   

## Motivation    

+ 기존의 GAN에서는 Generator가 실제 데이터 분포와 유사한 데이터를 생성하도록 하기 위해    
  Jensen-Shannon divergence나 Kullback-Leibler divergence 등의 분포 간 거리를 측정하는 방식을 사용.    

+ 하지만 이러한 방식은 학습의 불안정성과 mode collapse 문제를 일으키는 원인 중 하나가 됨.      

+ 기존 GANs의 training problem을 해결하기 위해 다른 loss function의 접근을 시도함.    

------------------------------------------------------------------------------------------------    

## Introduction    

+ 본 연구에서 관심을 가지는 문제는 비지도 학습이다.
{d}}
+ 확률 분포를 학습한다는 것은 무엇을 의미하는 것일까??
  - 고전적인 정답은 확률 밀도를 학습하는 것이다.
  - 이는 종종 밀도의 parametric family $(P_\Theta)$<sub>$\Theta\in\mathbb{R}^{d}$</sub> 을 정의함으로써 이루어지며, 우리 데이터에 대한 likelihood를 maximize하는 parameter를 찾게된다.
    > 개인적으로 이 $(P_\Theta)$<sub>$\Theta\in\mathbb{R}^{d}$</sub> 라는 것은 어떤 분포의 parameter일 수도 있고, 아니면 neural network의 가중치가 될 수 있는 것 같다.
     Gaussian distribution라고 한다면 평균과 표준편차가 parameter가 되고, 그래서 parametric family라는 용어를 사용한 것이 아닐까 싶다.  

+ 즉, 만약 우리가 실제 데이터 $\{ {x^{(i)}} \}_{i=1}^{m}$ 를 가지고 있을 때, 우리는 다음 문제를 해결하는 것이다.    

$$\max_{\theta\Theta\in\mathbb{R}^{d}}\sum_{i=1}^{m}\log P_\Theta \left ( x^{\left ( i \right )} \right ) $$       


+ 만약 실제 데이터 분포 $p_r$가 밀도를 나타내고 $p_\Theta$ 가 parametrized density $P_\Theta$의 분포라면,    
  점근적으로 이 양은 KL divergence $KL\left ( p_r || p_\Theta \right )$ 를 최소화한다.
  (즉, 실제 데이터 분포와 parameter로 나타내지는 밀도 사이의 거리를 최소화해서 parameter로 실제 데이터 분포에 가깝게 만들어보자는 의미이다.)       

+ 하지만, 이는 저차원 manifold에 의해서 support 받는 distribution을 다루는 일반적인 상황에서는 그렇지 않다.    

+ model manifold와 실제 분포의 support가 무시할 수 없는 교차점을 가질 가능성은 거의 없으며, 이는 KL divergence가 정의 되지 않는다는 것을 의미한다.    
  (support : 지지집합, 어떤 함수가 존재할 때 함숫값이 0이되는 정의역의 집합을 의미)   

+ 이러한 문제에 traditional한 방법은 model distribution에 noise term을 추가하는 것이다.
  > 고전적인 머신러닝 문헌에서 묘사되는 사실상 모든 생성 모델들이 noise component(Gaussian)를 포함하는 이유    

+ 존재하지 않는 $p_r$의 밀도를 추정하는 것 대신에, fixed distribution p(z)를 가지는 random variable $Z$를 정의할 수 있으며 이를 어떤 분포 $p_\Theta$를 따르는 sample를 직접적으로 만드는 parametric function $g_\theta : Z \rightarrow X$ (전형적인 어떤 종류의 신경망)에 통과시킨다.

+ $\theta$를 다르게 하면서, 우리는 이 분포를 변화시킬 수 있고 실제 데이터 분포 $p_r$에 가깝게 만들 수 있다. (Variational Inference)

+ 이는 두 가지 측면에서 유용하다.  
  > 1) 밀도와는 다르게, 이 접근법은 저차원의 manifold에 국한된 분포를 표현 가능   
  > 2) 쉽게 생성할 수 있는 능력은 밀도의 값을 아는 것보다 더 유용   
  ($\because$ 일반적으로, 임의의 고차원 밀도가 주어졌을 때 샘플을 생성하는 것은 연산적으로 어려운 일이다.)    

+ Variational Auto-Encoders (VAEs)와 GANs은 이러한 접근법으로 잘 알려져 있다.
  왜냐하면 VAE는 examples의 approximate likelihood에 초점을 두기 떄문에, 표준 모델의 한계점을 공유하며 추가적인 noise terms을 조작할 필요가 있다.    

+ GANs는 목적 함수의 정의에서 훨씬 더 많은 융통성을 제공하며, Jensen-Shannon과 모든 f-divergence, exotic combinations를 포함한다.    
  그러나, 이론적으로 연구된 이유들로 학습하기 까다롭고 불안정한 것으로 잘 알려져 있다.    

+ 본 논문에서는 거리나 분산 $\rho \left (p_\Theta,p_r \right )$ 를 정의하는 다양한 방법에 대해 모델 분포와 실제 분포가 얼마나 가까운지를 측정하는 법에 관심을 가진다.    

+ 이러한 distance/divergence 사이의 가장 근본적인 차이는 확률 분포의 sequence의 convergence에 미치는 영향이다.    
+ 분포의 sequence $\left (p_t\right)$<sub>$t\in\mathbb{N}$</sub> 는 $\rho \left (p_t,p_\infty \right )$ 가 0이 되는 경향이 있는 분포 $p_\infty$ 가 존재할 때 수렴하게 되며 이는 distance $\rho$가 얼마나 정확히 정의되는지에 달려있다.    

------------------------------------------------------------------------------------------------            

## Contribution    

본 논문의 contribution은 다음과 같다.    

+ 분포를 학습하는 관점에서 주로 사용되는 probability distance와 divergence를 비교하여 **Earth Mover(EM) distance**가 어떻게 작용하는지에 대한 포괄적인 이론적 분석을 제공한다.    

+ EM distance의 효율적이고 합리적인 approximation을 최소화하는 Wasserstein-GAN이라고 불리는 형태의 GAN을 정의하고, 대응되는 Optimization 문제가 타당하다는 것을 이론적으로 보인다.    

+ GAN의 주요한 학습 문제를 WGANs이 경험적으로 해결한다는 것을 보인다. 실제로 WGANs을 학습하는 것은 discriminator와 generator 사이의 조심스러운 균형을 유지하는 것이 요구되지 않으며 network architecture의 조심스러운 설계또한 요구되지 않는다. 
  GANs에서 주로 발생하는 mode dropping phenomenon 또한 매우 줄어든다. WGANs의 가장 주목할만한 실질적 이득은 discriminator를 학습시킴으로써 EM distance를 optimal까지 끊임없이 추정할 수 있는 능력이다.    
  이러한 학습 curve를 그리는 것은 hyperparameter search와 디버깅에 유용할 뿐만 아니라 관측되는 sample quality와 현저하게 correlation이 있다.    

------------------------------------------------------------------------------------------------ 

## Differnt Distances    
    

------------------------------------------------------------------------------------------------    

## Wasserstein GAN       



------------------------------------------------------------------------------------------------   

## Empirical Results    

+ 저자는 Wasserstein GAN algorithm을 사용하여 image generation에 대해서 실험을 수행하였으며, 표준적인 GAN에서 사용되는 formulation에 비해 WGAN을 사용하는 것이 상당한 이점을 가진다고 주장함.    
  - Generator 수렴과 샘플 품질과 상관관계가 있는 의미 있는 loss metric
  - 최적화 프로세스의 향상된 안정성    

------------------------------------------------------------------------------------------------     

### Meaningful loss metric   



------------------------------------------------------------------------------------------------    

### Improved Stability       


------------------------------------------------------------------------------------------------       


## Conclusion & Future Work(limitation)      



------------------------------------------------------------------------------------------------      

### 개인적으로 느낀 점   
     
     



