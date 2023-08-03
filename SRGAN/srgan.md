# [Photo-Realistic Single image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf)


------------------------------------------------------------------------------------------------     

## Motivation    

+ SR 분야에서 기존 방법들이 가진 단점들이 보이기 시작
  - 논문이 발표된 2017년을 기준으로 SR(Super-Resolution)분야에서 CNN이 다양하게 활용되고 있었다.
  - 하지만 Upsacle Factor가 큰 경우에 **미세한 질감**(**finer texture details**)을 복구하는데 어려움이 있다는 문제점이 존재했다
  - 그 이유는 바로 기존 모델들이 사용하는 **목적함수**(**Objective function**)에 있었다.  

+ 그렇다면 어떤 목적함수를 사용했길래??
  - 초기 SRCNN을 비롯한 대부분의 모델이 목적함수로 **MSE**(**Mean Squared Error**)를 사용했다      

  - **MSE**는 Pixel-wise 연산으로 **PSNR** 수치는 높게 얻을 수 있지만, 사람이 고화질이라고 느끼는 **지각적(Perceptual)** 감각을 전혀 표현할 수 없었다.      

  - 다시 말해, PSNR 수치가 더 낮게 나온 것이 화질이 더 좋아 보일 수 있는 것이다.

  - 또한, 고주파수 디테일(high-frequency details)의 표현이 상당히 떨어졌다.     
    >         
    > **PSNR**(**P**eak **S**ignal-to-**N**oise **R**atio) '최대 신호 대 잡음비'를 말하며       
    >         
    > 신호가 가질 수 있는 최대 전력에 대한 잡음의 전력을 뜻한다.      
    >        
    > 주로 영상 또는 동영상 손실 압충에서 화질 손실 정보를 평가할 때 사용된다.      
    >         
    > 손실이 적을 수록 (화질이 좋을수록) 높은 값.     
    >         
    > 반대로 무손실 영상의 경우, MSE가 0이 되기 때문에 PSNR을 정의할 수 없다.      
    >        
    > PSNR은 신호의 전력에 대한 고려 없이 **MSE**를 이용해서 계산할 수 있다.      
    >         
    > $$PSNR = 10 \cdot \log_{10} \left \( \frac{MAX_{I}^{2}}{MSE} \right \) = 20 \cdot \log_{10} \left \( \frac{MAX_{I}}{\sqrt{MSE}} \right \) = 20 \cdot \log_{10} \left \( MAX_{I} \right \) - 10 \cdot \log_{10} \left \( MSE \right \)$$     
    >         
    > $MAX_I$ : Pixel의 최댓값,  8bit unsigned integer 인 경우 255가 되고, double 이나 float type으로 소수점 형태로 표현될 경우 1      

<p align='center'><img src = 'https://github.com/WestChaeVI/GAN/assets/104747868/db5df738-a430-416f-acf2-  
edf76fb2bf55'><p>      

+ 위 이미지에서 PSNR과 MSE 기반 모델의 단점을 볼 수 있다.    

+ PSNR 수치는 SRResNet이 가장 높지만 SRGAN의 이미지가 사람의 눈에는 보다 더 고화질 이미지처럼 보인다.    

+ 또한, 목 부분을 보면 원본 이미지의 미세한 질감을 SRGAN 이외의 모델은 전부 부드럽게 뭉개서 표현하고 있는 것 처럼 보인다.     

+ 이는 MSR 자체가 **Pixel - Wise 평균**을 계산하기 때문에 발생하는 현상이다.   

------------------------------------------------------------------------------------------------    

## Introduction    

### MSE 기반 학습의 문제점 : 사람의 지각적 감각을 재현하지 못하다.    

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/4aec867b-5173-4a43-9f68-b083f41e70a4'><p>      

+ 오른쪽의 이미지는 왼쪽의 고해상도 임지를 오른쪽으로 한 칸(1 pixel) 평행 이동한 이미지이다.   

+ 인간이라면 두 이미지를 보고 해상도를 평가할 때, 두 이미지 다 해상도가 비슷하다고 평가할 가능성이 농후하다.    

+ 왜냐하면 단순히 평행 이동을 했을 뿐이기 때문이다.   

+ **그러나, MSE 기반의 학습된 모델의 입장에서는 그렇지 않다.**     

+ MSE는 같은 위치의 픽셀끼리의 차이를 계산한다. 따라서 모델은 평행 이동을 한 두 이미지의 해상도가 전혀 다르다고 판단하게 된다.   

+ 그래서, MSE를 활용하여 SR 모델을 학습하면 **Smoothing** 현상이 발생함.      
  > Smoothing 현상 : 이미지의 고해상도 세부 정보가 부분적으로 손실되거나 흐려지는 것을 말함.    
      
       

### 사람의 감각을 공감하고 재현할 수 있는 방법을 제안하다.    

+ 따라서 저자들은 이러한 문제들을 해결하기 위해 새로운 architecture와 Loss function, 그리고 test method를 제안안했다.    

#### Novel Architecture : GAN을 활용한 SR 모델, *SRGAN*     

+ GAN을 활용하여 "4x Upscaling"이 가능한 최초의 Framework를 만들어 냈다.     
  - SR 분야에 GAN을 활용하면 원본 이미지의 미세한 질감(detail texture)을 잘 보존한 고해상도 이미지를 만들 수 있다는 장점이 있다.      

+ 구조적으로는 Batch Normalization, Residual Blocks, Skip Connection을 활용한다.
  - 심층 신경망은 모델의 깊이가 깊어질수록 Loss가 사라지는 단점이 있는데 skip Connection을 활용하면 정보를 모델의 깊은 곳까지 전달 할 수 있는 장점이 있다.    


#### Novel Loss function : 지각적 손실 함수 (Perceptual Loss function)   

+ Perceptual loss function은 사람이 느끼는 고해상도 이미지와 가까워지기 위해 고안된 손실함수이다.    
  - "Adversarial Loss"와 "Content Loss"의 가중치 합(Weighted Sum)으로 구성된다.   

+ **Adversarial Loss**는 GAN에서 Generator의 Loss이다.
  - G는 진짜 같은 가짜 이미지를 만들어서 Discriminator를 속이도록 학습된다.    
  
  - 따라서 Adversarial Loss를 Total Loss에 포함시키면 생성하는 SR 이미지가     
    실제 이미지의 data space 근처에서 생성되도록 push할 수 있다.       

+ **Content Loss**는 MSE의 pixel 단위의 유사성을 지각적 유사성(Perceptual similarity)으로 대체하기 위해 제안된 손실 함수이다.  
  - 논문에서 Classification task를 위해 pre-trained VGG19 모델의 가중치를 활용했다.
    > pre-trained model을 사용한 이유는 기존 데이터의 **representation**을 잘 담고 있기 때문.    

  - G가 만든 가짜 고해상도 이미지와 진짜 고해상도 원본 이미지를 pre-trained model에 통과시키면 최종 결과물로 Feature map을 얻게된다.

  - 이 2개의 feature map 간의 차이를 구하는 것이 바로 content loss이다.    


+ 따라서 Adversarial Loss와 Content Loss의 Weighted sum으로 구성된 Perceptual Loss는 실제 고해상도 이미지를 데이터 공간 근처에서 원본 표현을 잘 따라 하는 SR 이미지를 만들 수 있게 도와주는 함수가 된다.     


#### Novel Test method : MOS (Mean Opinion Score)

+ 기존 SR 분야에서 사용되었던 PSNR이나 SSIM은 MSE 기반의 계산 방식이기 때문에 이미지 성능 평가에 적합하지 않다.    
+ 그래서 저자들은 MOS 테스트를 제안한다.    
  - MOS는 26명의 사람에게 1점(bad) ~ 5점(excellent)까지 점수를 매기도록 한 것이다.    
    > 저자들이 처음 제안한 것은 아니고 다른 분야에서 활용되고 있는 지표를 SR 분야에 새롭게 도입한 것이다.   

  - SSIM이란?    
    > SSIM(Structural Similarity Index Map)은 PSNR과 다르게 수치적인 error가 아닌 인간의 시각적 화질 차이를 평가하기 위해 고안된 방법이다.    
    >       
    > SSIM은 **Luminance, Contrast, Structural** 3가지 측면에서 품질을 평가하게 된다.   
    >       
    > $$SSIM \left \( x,y \right \) = \left \[ l\left \( x,y \right \)\right \]^{\alpha} \cdot \left \[ c\left \( x,y \right \)\right \]^{\beta} \cdot \left \[ s\left \( x,y \right \)\right \]^{\gamma}$$    
    >      
    > $$l\left \( x,y \right \) = \frac{2\mu_x\mu_y + C_1}{\mu_{x}^{2} + \mu_{y}^{2} + C_1} \ \ , \ \ c\left \( x,y \right \) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_{x}^{2} + \sigma_{y}^{2} + C_2} \ \ , \ \ s\left \( x,y \right \) = \frac{\sigma_{xy} + C_3}{\sigma_{x}\mu_{y} + C_3}$$      
    >       
    > <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/54a617e8-139e-4d9e-a56f-528976af5910'><p>    
    >      
    > Dynamic Range value인 L은 uint8 일때 0\~255의 값이며, default로 255를 사용하고 0\~1 사이로 표현된 값인 경우, default로 1이 사용된다.     

------------------------------------------------------------------------------------------------  

## Method    

### Trainset은 어떻게 구성하는가?

+ **SISR**(**S**ingle **I**mage **S**uper-**R**esolution) 분야의 목적은 저해상도 이미지로부터 고해상도 이미지를 추정하는 것이다.   
  - 따라서, trainset에는 low-resolution(lr), high-resolution(hr) 이렇게 한 쌍의 이미지가 필요

+ lr image는 hr image에 Gaussian filter를 적용한 다음 가로, 세로를 r배 (r : downsampling factor) 줄임으로써 얻을 수 있다.   
  - 즉, 원본 hr image의 size가 C x rH x rW라고 한다면, 생성된 lr image의 크기는 C x H x W 가 된다.  

### SRGAN의 목적함수는 어떻게 되는가?   

#### Discriminator Loss    

<table align='center'>
  <td>
    <p align='center'> $$\min_{\theta_G}\max_{\theta_D} \ \mathbb{E}_{I^{HR} \sim p_{\text{train}} \left( I^{HR} \right) } \left[ \log {D_{\theta_D}}\left( I^{HR} \right) \right] + \mathbb{E}_{I^{LR} \sim p_{G} \left( I^{LR} \right) } \left[ \log \left(1 - D_{\theta_D} \left( G_{\theta_G}\left( I^{LR} \right) \right) \right) \right]$$     
    </p>
  </td>
</table>    

+ 진짜 고해상도 이미지 $I^{HR}$ 이 Discriminator에 입력되면 $\log \ {1} \ = \ 0$ 이 되어야 한다.   
+ 반대로 가짜 이미지는 $\log \ {\left \(1 \ - \ 0 \right \)} \ = \ 0$ 이다.  
+ 따라서 **Discriminator**는 최댓값이 0인 목적함수를 0에 수렴하도록 **maximize**해야 한다.

#### Generator Loss    

<table align = 'center'>
  <th>
    <p align='center'>Perceptual Loss</p>
  </th>
  <th>
    <p align='center'>Adversarial Loss</p>
  </th>
  <th>
    <p align='center'>Content Loss</p>
  </th>
  <tr>
    <td>
      <p align='center'> $$L^{SR} = L_{X}^{SR} + {10^{-3}} \cdot L_{G}^{SR}$$     
      </p>
    </td>
    <td>
      <p align='center'> $$L_{G}^{SR} = \sum_{n=1}^{N} -\log D_{\theta_D} \left( G_{\theta_G} \left( I^{LR} \right) \right)$$     
      </p>
    </td>
    <td>
      <p align='center'> $$I_{\text{VGG}/i,j}^{\text{SR}} = \frac{1}{W_{i,j}H_{i,j}} \sum_{x=1}^{W_{i,j}} \sum_{y=1}^{H_{i,j}} \left( \phi_{i,j} \left( I^{HR} \right)_{x,y} - \phi_{i,j} \left( G_{\theta_G} \left( I^{HR} \right) \right)_{x,y} \right)^2  $$     
      </p>
    </td>
  </tr>
</table>     

+ Generator의 목표는 Content Loss와 Adversarial Loss의 가중치 합으로 구성된 Perceptual Loss를 최소화하는 것이다.   

+ 먼저, Adversarial Loss는 일반적인 GAN에서 생성자의 Loss 함수 공식과 동일하다. 
  - 형태를 보면 Discriminator의 목적함수에서 뒷부분만 차용한 형태이다.   
  - Generator는 진짜 같은 가짜 이미지를 만드는 것이 목표이므로 Discriminator의 목적함수에서 앞부분은 필요가 없다.   
  - 여기까지의 내용을 아무 의문점 없이 읽었다면 GAN에대해서 잘 알고 있거나 아니면 이해를 제대로 못한 것이다.
  - **수식을 보면, 왜 $\log \left( 1 - D_{\theta_D} \left( G_{\theta_G} \left( I^{LR} \right) \right) \right)$ 가 아니라 $-\log D_{\theta_D} \left( G_{\theta_G} \left( I^{LR} \right) \right)$ 인지에 대해 의문을 가져야 한다.**     

<table align='center'>
  <th>
    <p align='center'>$$\log \left( 1 - D_{\theta_D} \left( G_{\theta_G} \left( I^{LR} \right) \right) \right)$$</p>
  </th>
    <th>
      <p align='center'>$$-\log D_{\theta_D} \left( G_{\theta_G} \left( I^{LR} \right) \right)$$</p>
    </th>
    <tr>
    <td>
      <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/d9b45253-a967-46b3-b78b-3b3e4af76ad4'>    
      </p>
    </td>
    <td>
      <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/0e661bf0-2f13-4a7f-af8a-1ed5cf7573b6'>    
      </p>
    </td>
  </tr>
</table>  

+ GAN의 Adversarial Loss    

  수식을 편하게 쓰기 위해서 general하게 설명하겠다.    

  - $D \left( G \left(z \right) \right)$ 는 0~1 값을 가지고 $D \left( G \left(z \right) \right)$ 가 1에 수렴할 때 Loss가 최소가 된다.   
  
  - **문제점 : 학습 초반에 $G \left(z \right)$ 가 생성해내는 가짜 이미지가 형편없다 보니 D가 확실하게 가짜 이미지라고 판별하게 된다.**    
    > 즉, $D \left( G \left(z \right) \right) = 0$     
    >      
    > 자 loss함수에 등장하는 $\log \left( 1 \ - \ D \left( G \left(z \right) \right) \right)$ 의 그래프를 그려보자. (위 그림 왼쪽)   
    >     
    > $\log \left( 1 \ - \ D \left( G \left(z \right) \right) \right)$ 그래프에서 **$D \left( G \left(z \right) \right)$ 가 학습 초반인 0 근처일 때 학습하기에 기울기가 매우 작다.**    
    >     
    > **다시 말해, 학습 초반부터 학습이 잘 되지 않는 문제점이 발생한다.**

  - **관점 변경 : 기울기의 절댓값을 더 크게 만들어 초반의 안좋은 상황을 G가 빨리 벗어날 수 있도록 하고 싶다.**   
    >     
    > $\log \left( 1 \ - \ D \left( G \left(z \right) \right) \right)$ 대신에 $-\log D_{\theta_D} \left( G_{\theta_G} \left( I^{LR} \right) \right)$ 로 바꿔서 계산한다.   
    >     
    > 위 그림의 오른쪽을 보면 0 근처에서의 기울기가 무한대에 가까워 빨리 벗어날 수 있다.  
    >     
    > **결국 같은 문제를 관점을 달리해서 해결해 나아간 것이다.**   

+ 다음으로 Content loss는 SRGAN에서 Generator의 목적함수에 추가된 특별한 공식이다.    
  - $$I_{\text{VGG}/i,j}^{\text{SR}} 에서 VGG가 들어간 이유는 pre-trained model을 쓴 것을 표현한 것이다.
  - G가 생성한 가짜 이미지와 원본 진짜 이미지(HR)를 pre-trained VGG19 model에 통과 시킨다. 물론 fc-layer 이전까지   
  - 그러면 각각의 Feature map들을 얻을 수 있고, 이 Feature map끼리의 MSE 이다.
  - element-wise 연산 후 제곱하여 전체 원소 개수로 평균을 구하는 형태이다.  
  - Introduction에서 말했듯이 **간극(gap)을 줄이기 위해 G가 진짜 데이터의 표현을 잘 따라 하도록 학습이 진행된다.**   

------------------------------------------------------------------------------------------------    

## Model Architecture    


------------------------------------------------------------------------------------------------     

## Experiments    



------------------------------------------------------------------------------------------------      

## Conclusion & Future Work(limitation)      



------------------------------------------------------------------------------------------------      

### 개인적으로 느낀 점   
     



