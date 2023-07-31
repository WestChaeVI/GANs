# Unspurvised Representation Learning With Deep Convolutional Generative Adversarial Networks(ICRL 2016)    
[paper](https://arxiv.org/pdf/1511.06434.pdf)
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


------------------------------------------------------------------------------------------------    

## Empirical Validation    


------------------------------------------------------------------------------------------------    
