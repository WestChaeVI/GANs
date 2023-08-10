# [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (ICCV 2017)](https://arxiv.org/pdf/1703.10593.pdf)     

------------------------------------------------------------------------------------------------------------------------------      

## Motivation    

+ 기존 GANs들은 pair data를 이용해서 학습. **하지만, 실제로는 pair보다 unpair data가 훨씬 많음**.    
+ pair dataset을 준비하는 것은 시간과 비용이 매우 크다는 문제점 존재    

+ 어떻게 하면 unpair dataset으로 학습을 할 수 있을까 생각 $\rightarrow$ CycleGAN 제안     

------------------------------------------------------------------------------------------------------------------------------      

## Introduction    

+ Image-to-image translation : **pair image**을 이용해 input image와 output image 간의 mapping을 학습    

+ 저자는 논문에서 **unpaired image-to-image translation** 방법을 제안함.     
  - 하나의 이미지 collection에서 특징을 잡아내고 다른 이미지 collection으로 해당 특징을 어떻게 녹여낼 수 있을지 찾아냄 (어떠한 pair image 사용 x)    
  - lack of supervision in the form of paired examples $\rightarrow$ exploit supervision at the level of sets    
    > given one set of images in domain $X$ and a different set in domain $Y$     

+ 기존의 GANs의 training 방식은 mapping $G : X \rightarrow Y$ 를 학습하는 것이다. 그러나, 양방향 translation에는 적합하지 않다.    
  - **이러한 mapping $G : X \rightarrow Y$ 방식으로의 translation은 각각의 input $x$와 output $y$를 의미 있는 방향으로 짝을 짓는 것을 보장하지 않는다.**    
    > 기존 GAN의 objective는 domain 간의 translation을 학습하는 방향인 것이지, pairwise하게 이루어지는 것은 아니기 떄문이다.   
  - mode collapse 등의 최적화 문제 발생    
    > 모든 input 이미지가 하나의 output 이미지로만 mapping되고 더 이상의 최적화가 이루어지지 않는 점)     

+ The Concept of **Cycle consistent**    
  - 논문에서 예를 들기를 문장을 번역할 때, 영어 문장을 불어로 변역했을 때, 그 불어 문장을 다시 번역해도 본 영어 문장이 나와야한다는 말을 언급.    
  - 이를 수학적으로 접근하자면 두 개의 translator $G$, $F$가 있다고 가정하자.    
    이떄 $G$는 $X \rightarrow Y$이고, $F$는 $Y \rightarrow X$이다. $G$와 $F$는 서로의 **inverse**가 되어야 하며, 각 mapping은 bijection이어야 한다. (즉, 일대일 대응)    
  - Mappinf $G$와 $F$를 동시에 학습하고 $F(G(x)) \approx x$ 그리고 $G(F(y)) \approx y$ 를 만족하도록 하는 **Cycle consistency loss**를 추가한다.   
  - 이 Loss를 domain $X$와 $Y$에 대한 adversarial loss와 결합하면 unpaired image translation을 수행할 수 있다.     

------------------------------------------------------------------------------------------------------------------------------      
