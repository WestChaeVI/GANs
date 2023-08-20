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

## Related work    

+ GANs : adversarial training을 통해서 실제에 가까운 fake 이미지를 생성 $\rightarrow$ CycleGAN에서는 translation의 결과가 실제 이미지와 식별 불가능할 정도의 quality가 나오도록 하는 측면에서 사용할 수 있다.    

+ Image-to-Image Translation : paired image dataset 기반의 연구가 대부분이었으나, CycleGAN에서는 paired training example을 사용하지 않고 mappinf을 학습시킨다는 점에서 의의가 있다.     

+ Unpaired Imageo-to-Image Translation : unpaired setting에서의 image-to-image translation에 대한 연구 또한 이루어져 왔다. (두 domain을 relate시키는 방식)     
  1. Rosales et al. : **Patch 기반의 Markov random field**를 기반으로 한 **prior을 포함하는 Bayesian framework**    

  2. CoGAN and cross-modal scene networks : domain 간의 **공통적인 representation**을 학습하기 위한 **Weight-sharing strategy**     

  3. Liu et al. : Variational Autoencoder(VAE)와 GAN의 Combination     

  4. Method of encouraging the input and output to share speific "content" features even though they may differ in "style" : 이 방법 또한 adversarial network를 사용하지만, 미리 정의된 metric space(class label space, image pixel space, image feature space)에서 output과 input이 근접하도록 강제하는 term을 추가한 방법    

#### $\rightarrow$ 반면, CylceGAN에서는     

+ 어떠한 특정 task나, input과 output 간의 미리 정의된 similarity function에 의존하지 않는다.     

+ 기존의 방식처럼 input과 output이 동일한 저차원 embedding 공간에 있어야 한다고 가정하지 않는다.    
> 즉, 보다 유연하고 가정하는 사항들로부터 자유롭다. 이처럼 CycleGAN은 기존의 연구보다 vision task에서 일반적인 solution이 될 수 있다는 점에서 큰 의의가 있다.    

+ Cycle Consistency   

  -  구조화된 데이터를 regularization하는 방법으로 **transitivity**을 사용한다는 Idea는 오랜 역사를 가지고 있다.    

  - **back translation and reconciliation** : langauge domain에서 번역을 확인하고 개선하는 방법    
  - Zhou et al. & Godard et al. : cycle consistency loss 사용. (to supervised CNN training)     

  - Yi et al. (DualGAN) : Unpaired iamge-to-image translation을 위한 similar objective를 독립적으로 사용. (두 개의 generator가 존재하고, cycle consistent하게 해주기 위해 각 generator가 reconstruction loss를 최소화하는 방식으로 학습함)     

#### $\rightarrow$ CylceGAN에서도 cycle sonsistency loss를 사용해서 $G$와 $F$가 서로 consistent하도록 해준다.        

+ Neural Style Transfer : image-to-image translation을 수행하는 방법으로, 하나의 content image를 style image와 결합해서 새로운 이미지를 합성해내는 방법. pretrained된 deep feature의 Gram matrix 통계량을 구한 후 이를 매칭하는 방식으로 결합이 이루어진다.    

#### $\rightarrow$ CylceGAN에서는 두 개의 specific한 이미지 간의 mapping을 학습하는 것이 아니라 image collection 간의 mapping을 학습하는 것.     

#### 좀 더 higher-level의 appearance structure 간의 대응을 capture하는 방식으로 수행할 것.     즉, 보다 다양한 task에 적용할 수 있다는 이점이 있다.     

------------------------------------------------------------------------------------------------------------------------------      

## Formulation    

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/5ae19967-c17c-4bbe-8610-67cd138cad19'></p>      

+ notation     
  - $p_{data}\left(x\right)$ : $x$의 data distribution      
  - $p_{data}\left(y\right)$ : $y$의 data distribution      
  - $G$ : a mapping of $X \rightarrow Y$       
  - $F$ : a mapping of $F \rightarrow X$       
  - $D_{X}$, $D_{Y}$ : two adversarial discriminator       
    > $D_{X}$ : $x$와 $F\left(y\right)$를 구분     
    > $D_{Y}$ : $y$와 $G\left(x\right)$를 구분     

+ 문제 정의    
  - unpaired dataset을 가지고 image-to-image translation을 수행해야됨     
  - 두 개의 mapping $G$, $F$ (inverse of each other)      
    > 하나의 mapping은 제약이 매우 적기 때문에 **inverse mapping**을 하나 더 정의     
  - cycle consistent하도록 하는 cycle consistency loss term 정의    
    > 학습된 mappings $G$, $F$가 서로 모순되지 않도록 방지하는 역할     
  - 두 개의 adversarial loss + 한 개의 cycle consistency loss 가 필요      

------------------------------------------------------------------------------------------------------------------------------      

### 1. Adversarial Loss    

+ Generator $G$     

$$\mathcal{L}\_{GAN}(G,D_Y,X,Y) = \mathbb{E}\_{y \sim p_{data}(y)} \left\[log\ D_Y(y)\right\] + \mathbb{E}\_{x \sim p_{data}(x)} \left\[log\ (1 - D_Y(G(x))\right\]$$       

+ Generator $F$     

$$\mathcal{L}\_{GAN}(F,D_X,Y,X) = \mathbb{E}\_{x \sim p_{data}(x)} \left\[log\ D_X(x)\right\] + \mathbb{E}\_{y \sim p_{data}(y)} \left\[log\ (1 - D_X(F(y))\right\]$$       


### 2. Cycle Consistency Loss    

+ Adversarial training은 이론적으로는 mapping $G$와 $F$를 output generating function이 data distribution을 완벽히 근사할 수 있도록 학습시키는 게 가능함.     

+ 그러나 이는 모델 용량이 충분히 크다고 가정할 때 network가 input image dataset을 target domain의 임의의 이미지 permutation으로 mapping 됨을 의미한다.    

+ 즉, **adversarial loss는 학습된 함수가 각각의 input image $x_i$가 원하는 해당 output image $y_i$로 mapping되는 것을 보장하지 않는다.**    

+ 따라서, CycleGAN의 저자는 가능한 mapping 함수의 공간을 더 줄이기 위해서 학습된 mapping 함수가 "**cycle-consistent**" 해야 함을 주장한다.    

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/708d274f-c12d-4dcb-9162-81ff08be3f40' width='500'></p>     

+ **Cylce Consistency**    
  - $x \rightarrow G(x) \rightarrow F(G(x)) \approx x$의 cycle을 보장하기 위해서, $x$와 $x'(=F(G(X))$ 간의 간극을 줄이는 term이 cycle consistency loss이다.    
    > 이는 $y \rightarrow F(y) \rightarrow G(F(y)) \approx y$에도 똑같이 적용이 된다.    

  - cycle-consistency loss를 이용해서 최적화가 제대로 이루어지면 $G$, $F$는 cycle-consistent한 방식으로 mapping이 학습될 것이다.     

+ **Loss term**    
$$\mathcal{L}\_{cyc}(G,F) = \mathbb{E}\_{x \sim p_{data}(x)}\left\[\lVert F(G(x)) - x \rVert_1\right\] + \mathbb{E}\_{y \sim p_{data}(y)}\left\[\lVert G(F(y)) - y \rVert _1\right\]$$

  > 간단히, mapping된 값과 실제 original image 간의 $L_1$ loss의 기댓값으로 정의    
  >       
  > 참고로, $L_1$ loss 대신 adversarial loss를 사용했을 때 성능의 향상이 관찰되지 않았다.     


### 3. Full Objective     

$$\mathcal{L}\_(G, F, D_X, D_Y) = \mathcal{L}\_{GAN}(G, D_Y, X, Y) + \mathcal{L}\_{GAN}(F, D_X, Y, X) + \lambda \mathcal{L}_{cyc}(G,F)$$

+ 여기서 $\lambda$는 **cycle consistency loss term**과 **adversarial loss term** 간의 상대적인 중요도를 **control**하는 **hyper-parameter**이다.   

+ 따라서 최적의 $G$와 $F$는 다음과 같이 정의된다.    
  - $$G^{\*}, F^{\*} = arg \min_{G,F} \max_{D_X, D_Y} \mathcal{L}(G,F,D_X,D_Y)$$


#### AutoEncoder(AE)관점에서의 CycleGAN     

+ 논문에서 제시한 모델은 두 개의 auto-encoder를 학습하는 것으로도 볼 수 있다.     
  > 이는 각 mapping을 합성할 경우 결국에는 자기 자신을 복원하는 형태이기 때문이다. (AE의 형태)    
+ 하지만, 주목할 점은 이러한 auto-encoder의 내부 구조를 보았을 때 intermediate representation이 다른 domain으로의 translation하는 형태라는 점이다.     

+ Adversal training을 거친다는 점에서 adversarial auto-eoncoder의 특징을 보인다고 할 수 있다.    
  > 보다 더 구체적으로 들어가면, CycleGAN의 경우 임의의 target distribution을 대응시키기 위해서 adversarial loss를 활용해서 autoencoder의 bottleneck layer를 학습시키는 형태의 adversarial auto-encoder라고 할 수 있다.    
  >       
  > 이 경우 $X$를 복원시키는 autoencoder의 target distribution은 $Y$가 된다.    
  >       
  > 즉, "Cycle" = reconstruction => autoencoder로 볼 수 있다. 다만 latent space(or traget distribution)이 target domain $Y$라는 점이다. 

------------------------------------------------------------------------------------------------------------------------------      

## Implementation     

### Network Architecture    

#### Generator     

  + 3개의 convolution layer    
  + several residual blocks    
  + 2 fractionally-strided convolutional with stride = $\frac{1}{2}$    
  + one covolution that maps features to **RGB**     
  + 6 blocks for 128 by 128 images and 9 blocks for 256 by 256 or higher-resolution images     
  + instance normalization    

#### Discriminator     

  + 70 by 70 **PatchGANs** : 70 by 70 크기의 overlapping 이미지 patch를 구성하고, 해당 패치에 대한 진위여부를 판별하는 구조     

  + 이러한 Patch-level discriminator은 전체 이미지를 입력으로 받는 discriminator에 비해 적은 파라미터를 갖고, fully convolutional한 구조를 가져서 임의의 사이즈의 image에도 대응이 가능     

### Training Details     

+ 학습 안정성 제고를 위한 2가지 방법    
  - Mod to adversarial Loss $L_{GAN}$   
    > Negative Likelihood Loss (NLL)이 아닌, **Least-squares Loss**를 사용    
    >       
    > 학습 과정 중 보다 안정적인 양상을 보여주고, 보다 양질의 이미지를 생성할 수 있었다.     
    >       
    > train $G$ to minimize $\mathbb{E}_{x \sim p\_{data}(x)}\left\[(D(G(x)) - 1 \right\]^2$   
      > 가짜를 진짜로 판별할 확률과 1값의 차이를 줄여나간다.      
      > 즉, 가짜를 진짜로 판별할 확률을 maximize하는 것과 다름 없다.     
    >         
    > train $D$ to minimize $\mathbb{E}_{y \sim p\_{data}(y)}\left\[(D(y) - 1 \right\]^2 + \mathbb{E}\_{x \sim p{data}(x)}\left\[D(G(x))\right\]^2$      
      > 진짜를 진짜로 잘 판별해내고 (앞의 term)    
      > 가짜를 진짜로 판별해낼 확률을 줄여감 (뒤의 term)    

  - 가장 최근의 generator가 아니라, 모든 생성된 이미지의 history를 다 이용해서 discriminator를 학습시킨다.    
    > 이를 위해 최근 생성된 이미지 50개를 저장할 수 있는 이미지 버퍼가 필요    

+ Hyper-Parameter    

  - \lambda = 10 으로 설정    
  - Adam solver 사용   
  - batch size = 1     
  - learning rate = 0.0002   
  - 100 epoches 학습 후 나머지 100 epoches는 점점 learning rate를 0에 가깝게 줄여나간다.    


------------------------------------------------------------------------------------------------------------------------------      

## Results     

+ 최근의 unpaired image-to-image translation 다른 방벙론들과의 비교(on paired datasets)를 통해 모형의 성능 평가    

+ Loss에 대한 ablation study를 통해 adversarial loss와 cycle consistency loss의 중요도를 평가   
  > ablation study : model이나 argorithm을 구성하는 다양한 구성요소 중 어떠한 feature를 제거할 때, 성능에 어떠한 영향을 미치는지 파악하는 연국 방법      
   
+ paired dataset이 없을 경우의 알고리즘 활용성을 위한 알고리즘 확장성(generality)에 대한 평가   

### Evaluation   

Pix2pix와 동일한 방법으로 quantitative, qualitative 평가를 수행하고 이를 여러 baseline과 비교    

+ tasks    
  - semantic labels $\leftrightarrow$ photo on the Cityscapes dataset   
  - map $\leftrightarrow$ aerial photo on data scraped from Google Maps    
  - ablation study on the full loss function   

### Evaluation Metrics   

+ AMT perceptual studies (qualitative)   
  - map $\leftrightarrow$ aerial photo에 대해서 적용    
  - 25명의 참가자에게 이미지가 진짜인지 아닌지 판단하도록 함.    
  - baseline과 비교를 수행    

+ FCN score (quantitative)   
  - Cityscapes labels $\rightarrow$ photo task    
  - FCN은 생성된 사진에 대해 label map을 예측해서 input의 GT labels와 비교   
    > 즉, label값을 바탕으로 $G$는 사진을 생성하고 이를 다시 FCN을 통과해서 label을 예측.    
    > 생성이 잘 되었다면 동일한 label을 return.     

### Baselines     

+ CoGAN     
  - one GAN generator for domain $X$ and one for domain $Y$ with tied weights on the first few layers for shared latent representations     
  - 이미지 $X$를 생성하는 latent representation을 찾고 이를 $Y$의 스타일로 랜더링하는 형태    

+ SimGAN    
  - domain 간 translation을 위해 adversarial loss를 사용   
  - pixel level에서 큰 변화를 주는 것에 penalty를 주기 위해 regularization term $\lVert x - G(x) \rVert_{1}$ 를 더해준다.    

+ Feature Loss + GAN    
  - a variant of SimGAN (RGB pixel value가 아닌, pre-trained network에서 image features를 추출해서 $L_{1}$ Loss가 계산됨 $\rightarrow$ perceptual loss    

+ BiGAN / ALI   
  - 원래는 unconditional GANs with inverse mapping    
  - 실험에서는 conditional setting으로 임의로 바꿔 성능 비교 진행    

+ Pix2pix    
  - trained on paired dataset   
  - CycleGAN이 어떠한 paired data 없이 얼마나 upper bound에 근접할 수 있는지 확인   

변인통제를 위해 CoGAN을 제외하고는 backbone과 training detail은 모두 고정하여 실험 진행   
  > CoGAN의 경우, image-to-image network의 구조가 아니었기 때문에 환경을 통일할 수 없었다.    

### Comparison against baselines    

+ unpaired setting에서의 baseline은 모두 CycleGAN에 비교되는 성능을 보여주지 못했다.   

+ 반면, CycleGAN은 paired setting에서 학습된 pix2pix에 견줄 만한 성능을 보여주고 있다.    

<table align='center'>
  <th>
    <p align='center'>Citiyscapes test</p>
  </th>
  <th>
    <p align='center'>Map $\leftrightarrow$ area test</p>
  </th>
  <tr>
    <td>
      <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/b82070c1-9c83-496b-ae1b-4e03f607e449'></p>
    </td>
    <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/756e7df6-8abc-44bd-9eb4-500264297287'></p>
    </td>
  </tr>
</table>      

+ 성능 비교표    

  - 역시 다른 unpaired image-toimage translation 알고리즘보다 월등한 성능을 보여주고 있고, pix2pix에 가장 가까운 성능을 보여주고 있다.   

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/f29fdb25-97e5-427f-a87b-dd66618fee82' height='500'></p>      

### Analysis of the loss function      

+ full loss term에서 다양한 variants를 주면서 FCN score를 비교    

+ GAN Loss or Cycle-consistency loss를 제거하면 **성능의 심각한 저하**를 야기   
  > Cycle-consistency loss의 방향에 대한 실험을 진행한 결과, **labels to photo의 경우**에는 양방향 loss가 다 추가된 경우보다 **forward cycle만이 추가된** 경우가 조금 더 높았으나,     
  > 정성적으로 결과를 시각화해본 결과 **mode collapse**와 **training instability가 발생함**을 알 수 있었다.   

<table align='center'>
  <th>
    <p align='center'>Quantitative eval</p>
  </th>
  <th>
    <p align='center'>Qualitative eval</p>
  </th>
  <tr>
    <td>
      <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/3eaa6034-8817-4f50-9749-a4b9c09e6b86'></p>
    </td>
    <td>
    <p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/233333b0-a97b-449e-b9c9-b2bb767d60da'></p>
    </td>
  </tr>
</table>      


### Image reconstruction quality   

+ training과 test time 모두 reconstructed image가 original input과 종종 비슷    

+ map $\leftrightarrow$ aerial 의 경우에는 하나의 도메인(map)이 나머지 도메인(aerial)보다 훨씬 더 다양한 정보를 담고 있음에도 불구하고 어느 정도 복원이 잘 된 것을 알 수 있다.    

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/bd6fd999-2e82-4518-8215-a4cf19b4bdf5'></p>        

### Additional results on paired datasets     

+ CMP Facade dataset & UT Zappos50K (pix2pix train dataset)를 입력으로 한 생성 결과이다.   

+ supervised setting에서 학습된 모델과 거의 비슷한 결과 quality를 보여준다는 점이 인상적이다.   

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/6f3d8828-240e-4a37-803c-e6416b08a24c'></p>        

------------------------------------------------------------------------------------------------------------------------------      

## Limitations and Discussion     

+ 항상 stability를 유지하면서 좋은 결과를 보여주진 않았다. 종종 failure cas를 보여줌    

### Failure case 분석    

<p align='center'><img src='https://github.com/WestChaeVI/GAN/assets/104747868/0c85839f-13fb-4b68-be60-86ff349ffe8d'></p>   

+ color or texture의 change는 잘 학습 했으나, 비교적 큰 변화인 geometric change(i.e. form dog to cat)의 경우에는 제대로 학습이 되지 않았다.    
  > 논문에서는 모델 구조가 appearance change에 맞춰 구성되어 있다는 점을 이유로 추측하고 있다.     

+  appearance change의 경우에도, 학습 dataset의 distribution 특성에 따라 실패하는 경우가 종종 발생.    
  - 예를 들어, **horse to zebra와 같은 경우엔 trainset**이 **야생에서의 동물을 주로 담고 있기 때문**에 사람이 타고 있는 말 등의 모습은 제대로 다루지 못하는 경우가 발생.    

+ unsupervised learning으로서의 한계    
  - supervised setting에서의 translation newtork보다 성능 저하가 두드러졌다.    
  - 이러한 성능 차이는 해결하기가 매우 어렵다.     
    > weakly supervised or semi-supervised learning의 방법으로 해결 해볼 수 있다.    
  - 그럼에도 불구하고 unsupervised setting에서 image-to-image translation의 성능을 크게 높였을 뿐 아니라, 보다 다양한 application이 가능해졌다는 점에서 그 의의가 크다.     


------------------------------------------------------------------------------------------------------------------------------      

## 개인적으로 느낀 점 (생각해 볼 수 있는 점들)    

+ network의 backbone은 auto-encoder 대신 resnet을 사용했다.    
  - 이유     
    > **깊은 네트워크 구조와 feature 추출 능력**    
    > CycleGAN은 두 domain 간의 image translation을 수행하기 위해 고수준의 추상적인 feature를 학습해야 한다.    
    > Resnet은 이러한 feature를 효과적으로 추출할 수 있는 구조를 가지고 있어, 더 나은 translation quality를 가지도록 돕는다.    
    >       
    > **학습의 안정성과 수렴 속도 개선**      
    > Resnet과 같은 구조는 빠른 수렴과 안정적인 학습을 돕는 효과를 가지고 있는 모델이다.     
    > CycleGAN은 두 domain 간의 mapping을 동시에 학습하고 **cycle-consistent 를 유지하려고 노력하기 때문에, 안정적인 학습이 중요하다.**   
    >      
    > 즉, Autoencoder 구조보다 ResNet과 같은 구조가 CycleGAN의 목적인 image translation task에 더 적합하다고 생각해볼 수 있다.     
    > autoencoder의 경우 geometric change에 취약해진다.    

+ Inverse mapping은 reconstruction이다. 그렇다면, $X \rightarrow Y \rightarrow X$만 하면 되는데 왜 $Y \rightarrow X \rightarrow Y$까지 했을까?     
  - 이유    
    > **Cycle Consistency보장을 통한 학습 개선**     
    > $Y \rightarrow X \rightarrow Y$ 또한 Cycle consistency를 보장하며 학습하는데 도움이 된다.   
    > 두 도메인 사이의 mapping은 일반적으로 양방향으로 일관성을 유지해야 한다.       
    > 그렇지 않으면 두 도메인 간의 변환 과정에서 정보 손실이나 왜곡이 발생할 수 있다.   
    > $Y \rightarrow X \rightarrow Y$의 inverse mapping을 통해 domain $Y$에서 domain $X$로 변환한 후 다시 domain $Y$로 돌아와야 하므로, cycle-consistent를 항샹시키고 원본 데이터를 더 잘 보존하는 모델을 학습할 수 있다.    
    >      
    > 또한, augmentation + generalization 효과까지 줄 수 있어 성능을 더 높일 수 있다.    


