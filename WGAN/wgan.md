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

+ 확률 분포를 학습한다는 것은 무엇을 의미하는 것일까??
  - 고전적인 정답은 확률 밀도를 학습하는 것이다.
  - 이는 종종 밀도의 parametric family $(P_\Theta)$<sub>$\Theta\in\mathbb{R}^{d}$</sub> 을 정의함으로써 이루어지며, 우리 데이터에 대한 likelihood를 maximize하는 parameter를 찾게된다.
    > 개인적으로 이 $(P_\Theta)$<sub>$\Theta\in\mathbb{R}^{d}$</sub> 라는 것은 어떤 분포의 parameter일 수도 있고, 아니면 neural network의 가중치가 될 수 있는 것 같다.
     Gaussian distribution라고 한다면 평균과 표준편차가 parameter가 되고, 그래서 parametric family라는 용어를 사용한 것이 아닐까 싶다.  

+ 즉, 만약 우리가 실제 데이터 $\{ {x^{(i)}} \}_{i=1}^{m}$ 를 가지고 있을 때, 우리는 다음 문제를 해결하는 것이다.    

$$\max_{\theta\Theta\in\mathbb{R}^{d}}\frac{1}{m}\sum_{i=1}^{m}\log P_\Theta \left ( x^{\left ( i \right )} \right ) $$       


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

## Concept of Distance
+ WGAN 논문에서는 4개의 거리가 등장하는데, 수식들을 설명하기에 앞서 먼저 기초적인 것들부터 짚고 넘어가보자.

+ 실수($R$) or 복소공간($C$) 에서는 |ㆍ|이 metric이다.
+ 유클리드 공간 ($R^{n}$) 에선 Euclidian distance가 metric이다.  $d\left ( x,y \right ) = \sqrt{\left (\sum\limits_{k=1}^{n}{|x_k - y_k |^2}  \right )}$
+ 힐베르트 공간 (Hilbert space)에서는 내적(inner product)으로 metric을 정의. $d\left ( u,v \right ) = \sqrt{\left ( u-v \right ) \cdot \left ( u-v \right )}$      

+ 어떤 공간에 metric 개념이 중요한 이유는 **수렴(Convergence)** 이란 정의를 내릴 수 있기 때문이다.    
$$x_n \rightarrow  x   \Leftrightarrow    \lim_{n\rightarrow\infty}{d\left (x_n,x \right )} = 0$$       

+ But, 한 공간에 정의할 수 있는 metric은 한가지만 있는 것이 아니다.

+ 예를 들어, 유클리드 공간에서는 Euclidian distance 뿐만 아니라, 맨허튼 거리 같은 다른 metric으로 거리를 정의할 수 있다.   $d\left ( x,y \right ) = \sum\limits_{k=1}^{n}{|x_k - y_k |^2}$       

+ 함수 공간에서는 더욱 다양하게 정의할 수 있다.     
  - $L_{1}$ 거리 : $d_{1}{\left (f,g \right )} = \lVert f-g \rVert_{1} = \int_{x}{\|f(x) - g(x)\|}\ dx$      
  - $L_{2}$ 거리 : $d_{2}{\left (f,g \right )} = \lVert f-g \rVert_{2} = \sqrt{ \int_{x}{\|f(x) - g(x)\|^{2}}\ dx }$     

<table style="margin-left: auto; margin-right: auto;">
  <th>
    <p align='center'>$L_{2}$ Convergence</p>
  </th>
  <th>
    <p align='center'>$L_{\infty}$ Uniformly Convergence</p>
  </th>
  <tr>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/664e8110-12c7-4d72-9e46-cc6b5ed98557'>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/35c024e7-8443-44e8-9680-149f1b1a2bd4'>
      <p>
    </td>
  <tr>
    <td>
      <p align='center'>$f_n$과 $f$의 차이를 제곱해서 적분한 값이 0으로 수렴하게 만들 수 있으면 $L_{2}$ - 수렴<p>
      <p align='center'>$\lVert f_n-f \rVert_{2} = \sqrt{ \int_{a}^{b}{\|f_n - f\|^{2}}\ dx } \rightarrow 0$ (수렴)<p>
    </td>
    <td>
      <p align='center'>$f_n$이 $f$로 모든 $x$에 대해서 $\epsilon$ 범위 안에 들어오면서 수렴하게 만들 수 있으면 $L_{\infty}$ - 수렴 또는 균등수렴(uniformly convergence)<p>
      <p align='center'>$\lVert f_n-f \rVert_{\infty} =  \sup\limits_{x\in[a,b]}{\|f_n - f\|}  \rightarrow 0$<p>
    </td>
  </tr>
  </tr>
</table>    

+ 즉, 전달하고 싶은 말은 **거리함수가 바뀌면, 수렴방식이 바뀔 수 있다는 것**이다.    

+ WGAN 논문은 분포수렴과 동등한 Wasserstein distance를 다룬다. 이 metric은 **확률분포**들의 공간에서 정의된다.    

+ $\chi$ : Compact metric set    
  - Definition : A topology space $\chi$ is called compact if each of its open covers has a finite subcover.    
  > Heine - Borel 정리를 이용해 해석하면, $\chi$가 compact라는 것은 *1) 경계가 있고*, *2) 동시에 경계를 포함*한다는 집합이다.    

<table style="margin-left: auto; margin-right: auto;">
  <th>
    <p align='center'>1) 경계가 있다 (bounded)</p>
  </th>
  <th>
    <p align='center'>2) 경계를 포함한다 (closed)</p>
  </th>
  <tr>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/fda694eb-1df8-4922-b60c-45aae693df00' width=400, height=300>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/dbbc7f6d-d594-4ed5-bc56-94b4f29d1594'width=400, height=300>
      <p>
  <tr>
    <td>
      <p align='center'>경계가 있다는 것은 무한대로 뻗지 않는다는 것을 의미. 집합 안의 아무 점에서나 적당한 거리 안에 $\chi$의 모든 원소가 들어오면 된다.<p>
    </td>
    <td>
      <p align='center'>어떤 집합의 경계를 포함한다는 것은 풀어서 표현하면 극한점(limit point)을 모두 포함한다라고 설명한다. 즉, $x_{\infty}$ 같은 극한점들이 $\chi$의 원소라는 것.<p>
    </td>
  </tr>
  </tr>
</table>  

+ 저자들이 compact 집합을 가져온 것은 수학적인 이유 때문이다.    
  - 최대ㆍ최소 정리 : 연속함수들이 항상 max, min 값을 가진다.   
  - 모든 확률변수 x에 대해 조건부 확률분포가 잘 정의된다.   
  - 완비공간(complete space)이다.   

+ $\sum$ : set of all the **Borel subsets of $\chi$**
  - Borel set은 $\chi$ 내에서 측정가능한(measurable)한 집합들을 말한다.
  - 측정가능이라함은 $\mathbb{P}_r$ , $\mathbb{P}_g$ 같은 확률분포로 확률값이 계산 가능한 집이다. $\rightarrow$ 연속함수의 기댓값을 계산하기 위한 수학적인 최소조건이 된다.
  - 사실 눈으로 관찰할 수 있는 집합들은 대게 측정가능한 집합들이다.(**측정불가능한 집합들은 컴퓨터로 나타내는게 불가능**)      

+ $Prob\left (\chi \right )$ denote the space of probability measures on $\chi$

+ 참고로 논문을 읽다보면 $\inf$ 와 $\sup$ 이라는 용어들이 튀어나온다.
  - $\inf A $= max{lower bound of A}
  - infimum 은 the greatest lower bound 라고 부른다. 즉, 하한(lower bound) 중 가장 큰 값(maximum)
  
  - $\sup A $= min{upper bound of A}
  - Supremum 은 the least upper bound 라고 부른다. 즉, 상한(upper bound) 중 가장 작은 값(minimum)

+ 이 개념들이 필요한 이유는 모든 집합이 최소값(혹은 최대값)을 가지지 않지만 $\sup$ $\inf$ 는 **항상 존재하기 때문**이다.     
$$A = \left \[ {1} , \frac{1}{2} , \frac{1}{3} , ... \right \] \ \inf A = 0 \ , \ min \ A = \ \?$$


------------------------------------------------------------------------------------------------    

## Differnt Distances   

### Total Variation (TV distance)   

  - 빨간색 A의 영역 안에 있는 A들을 대입했을 때, $\mathbb{P}_r \left \( A \right \)$ 와 $\mathbb{P}_g \left \( A \right \)$ 의 값의 차이 중 가장 큰 값을 뜻함.    
  - Total Variation 은 두 확률측도의 측정값이 벌어질 수 있는 값 중 가장 큰 값(supremum) 을 말한다.    
    
$$\delta{ \left \( \mathbb{P}_r , \mathbb{P}_g \right \) } = \sup{A\in\sum}\|{\mathbb{P}_r \left \( A \right \) - \mathbb{P}_g \left \( A \right \)}\|$$     

  <p align='center'>
  <img src='https://github.com/WestChaeVI/GAN/assets/104747868/48f3b12e-c742-458d-a272-f0d583572501'>
  <p>       

  - 같은 집합 $A$ 라 하더라도 두 확률 분포가 측정하는 값은 다를 수 있다.      
  - 이때 TV는 모든 $A\in\sum$ 에 대해 가장 큰 값을 거리고 정의한 것이다.      
  - 만약 두 확률분포의 확률밀도함수가 서로 겹치지 않는다면, 다시 말해 확률 분포의 **support**의 교집합이 공집합이라면 TV 는 무조건 1 이다.      



### Kullback - Leibler (KL divergence)    

  - KL divergence는 정보 entropy를 이용해 두 확률분포의 거리를 계산한다.     
$$DKL\left \( \mathbb{P}_r || \mathbb{P}_g \right \) = \sum\{x \in X} {P_r \left \( x \right \)} \cdot  {\log \left \( \frac{P_r{\left \( x \right \)}}{P_g{\left \( x\right \)}} \right \)}$$        

  - 만약 두 분포의 support 사이의 거리가 멀면 KL term이 발산한다.
    <p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/1d9179b3-ee85-492f-9f3d-97d9d6cb57c7' width = 600, height= 300>
    <p>     

  - 그 이유인즉슨 $\frac{P_r{\left \( x\right \)}}{P_g{\left \( x\right \)}}$ 에서 분자는 0보다 큰 양수인데, 분모는 0이기 때문에 발산을 하게 된다.   


### Jensen - shannon (JS divergence)      

  - KL term은 Symmetry하지 않기 때문에 유사도를 이야기할 때 distance라고 표현하지 않는다.
  - 이 거리 개념을 Distance Metric으로 쓸 수 있는 방법에 대해 고민하면서 나온 개념이 바로 JS divergence 이다.
  - JS divergence는 KL term으로 표현할 수 있다.
$$JS \left \( \mathbb{P}_r || \mathbb{P}_g \right \) = \frac{1}{2}KL \left \( \mathbb{P}_r || \mathbb{P}_m \right \) + \frac{1}{2}KL \left \( \mathbb{P}_g || \mathbb{P}_m \right \) \ , \ where \ \mathbb{P}_m = \frac{\mathbb{P}_r + \mathbb{P}_g}{2}$$    

  - $P_m \ = \ 0$ 이면  $P_r \ = \ P_r \ = \ 0$ 이기 때문에 **발산할 일은 없다**.     
  - 하지만 두 분포의 support가 겹치지 않는다면   
    > $\mathbb{P}_g{\left \( x\right \)} \ \neq \ 0 \ \rightarrow \ \mathbb{P}_r{\left \( x\right \)} \ = \ 0$
    > $\mathbb{P}_r{\left \( x\right \)} \ \neq \ 0 \ \rightarrow \ \mathbb{P}_g{\left \( x\right \)} \ = \ 0$   이기 때문에
    <p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/71459317-6d42-460a-9a36-b5f6a2df25c8' width = 800, height= 200>
    <p>    
  - **발산하지는 않지만 상수인 $\log{2}$ 로 고정되어 버리니 "얼마나 먼지"에 대한 정보를 줄 수 없는 것이다.**
  - 이런 일이 일어나는 이유는 TV나 KL, JS 은 두 확률분포 $\mathbb{P}_r$ , $\mathbb{P}_g$ 가 서로 다른 영역에서 측정된 경우     
    '완전히 다르다'고 판단을 내리게끔 metric이 계산되기 때문이다.
  - $\rightarrow$ 즉, 두 확률분포의 차이를 명탐정처럼 깐깐하게(harsh) 본다는 것.     
  - 이게 상황에 따라 유리할 수도 있겠지만, GAN에서의 discriminator의 학습이 잘 죽는 원인이 된다. (Martin의 정리)  
  - 그래서 GAN의 학습에 맞게 조금 유연하면서도 수렴(convergence)에 focus를 맞춘 다른 metric이 필요. $\rightarrow$  **WGAN의 motivation**


### Eeath Mover (EM distance) = Wassertein       

  $$W\left \( \mathbb{P},\mathbb{Q}\right \) \ = \ \inf\limits_{\gamma\in\prod\left \( \mathbb{P},\mathbb{Q}\right \)}{\int d\left \( x,y\right \)\cdot\gamma\cdot\left \( dxdy\right \)} \ = \ \inf\limits_{\gamma\in\prod\left \( \mathbb{P},\mathbb{Q}\right \)} \mathbb{E}^{\gamma}\left \[ d(X,Y) \right \]$$

  - $\prod\left \( \mathbb{P},\mathbb{Q}\right \)$ 는 두 확률분포의 결합확률분포의 집합, $\gamma$ 는 그 중 하나
  - 즉, $\prod\left \( \mathbb{P},\mathbb{Q}\right \)$ 중에서 $d(X,Y)$ 의 기댓값을 가장 작게 추정한 값을 의미
  - 직관적으로, $\gamma \left \(x,y \right \)$ 는 분포 $\mathbb{P}_r$ 를 분포 $\mathbb{P}_g$ 로 변형시키기 위해서 얼마나 많은 "질량"을 x에서 y로 운송해야 하는지를 나타낸다.
  - EM distance는 결국 cost의 optimal를 추정하는 문제로 볼 수 있다.   

+ 다음 예시는 확률 분포의 간단한 sequence가 EM distance 하에서 얼마나 명백하게 수렴하고 다른 distance와 divergence 하에서는 수렴하지 않는지를 보여준다.    
  - (수평선을 학습하는 경우) $\mathbb{Z} / ~ / \mathbb{U} \left \[ 0,1 \right \]$ 는 unit interval 하에서 균이 분포이다. $\mathbb{P_0}$ 는 $\left \( 0,\mathbb{Z} \right \) \in \mathbb{R}^2$ (0은 x축이고 random variable $\mathbb{Z}$ 는 y축)의 분포라고 가정하고, 원점을 통과하는 직선의 수직선이다. 

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/9e0c627f-1d11-49c0-9091-4e839c28e8ca' >
<p>

  - $g_\theta \left \( z\right \) \left \( \theta, z \right \)$ 라고 하고 $\theta$는 single real parameter이다.

  - $\theta_t \ \rightarrow \ 0$ 일 때, sequence  $\left \( \mathbb{P}_{\theta_t} \right \)$<sub>$t\in\mathbb{N}$</sub>    는 distance 하에서 $\mathbb{P}_0$ 으로 수렴하지만, JS, KL, reverse KL, TV 하에서는 수렴하지 않는다.    
  아래 그림 **Figure 1** 은  EM과 JS 의 경우에 대해 이를 나타낸다.    

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/1e98b918-6cc3-4c55-a1e8-340f0eb03931' >
<p>     

  - Figure 1을 살펴보면, JS divergence의 경우 대부분의 케이스에서 값이 같으므로, 적절한 gradient 값을 얻기가 어렵다.
  - 반면에, EM distance의 경우 대부분의 케이스에서 유의미한 gradient 값을 얻을 수 있게 된다.     

  - Example 1은 EM distance에 대해서 gradient descent를 진행함으로써 저차원의 manifold에 대한 확률 분포를 학습할 수 있는 케이스를 우리에게 제공한다.
  - Wasserstein distance가 JS divergence에 비해서 훨씬 약하기 때문에, 가벼운 Assumption 하에서도  $W{ \left \( \mathbb{P}_r , \mathbb{P} _\theta \right \) }$ 가 $\theta$ 에 대한 연속형 loss function 인지 아닌지에 대한 질문을 던져볼 수 있다. **결론은 True이다.**

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/fea38990-6a67-4561-aeb8-36ee16a95f73' >
<p>     
<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/aaf32567-be95-4766-9fb1-aab91f6038a0' >
<p>      

  - Assuption 1에서 **locally Lipschitz**라고 하는 내용이 있다. [Lipschitz 설명](en.wikipedia.org/wiki/Lipschitz_continuity)   

  - Lipschitz function이라는 것은 얼마나 빨리 변화할지가 제한된 함수라는 것이다.     
  - 그리고 그 bound의 가장 가장 작은 값이 Lipschitz constant라는 것이다.   
  - 즉, 어떤 함수가 Lipschitz constant라는 값보다 항상 변화율이 작아야 하는 것.     

  - Assuption 1은 이 Lipschitz constant의 기댓값이 무한대보다 작은 경우 g라는 함수가 다음 가정을 만족한다고 적혀있다.
    > 변화율을 어느 특정 임계치보다 작게 한다는 것은 아무래도 gradient가 과하게 변하는 상황을 방지하는 의미 같다.    
    > 쓸모 있는 gradient를 얻을 수 있게 한다고 생각하면 될 것 같다.     

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/24918168-d78a-4007-9e31-a6928d187280' >
<p>      

  - $z$ 에 대한 prior $p(z)$ 에 대해서 이에 대한 distance의 기댓값이 무한대보다 작으면, 아까 얘기했던 locally lipschitz 조건을 만족시키게 되고, Wasserstein distance가 모든 곳에서 연속이고 거의 모든 곳에서 미분 가능하다는 내용이다.   

  - 이 모든 것들은 EM distance가 JS divergence에 비해서 우리 문제에 대해 훨씬 더 민감한 cost function이라는 것을 보인다.   

  - 다음 theorem은 이러한 distance와 divergence에 의해서 유도되는 위상의 상대적 강함을 묘사하며, KL이 가장 강력하고, 그 다음으로는 JS, TV, EM이 가장 약하다.  

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/138610d6-8dc6-4382-8661-2ef4246ff975' >
<p>    

  - 이는 KL, JS, TV distance가 low dimension의 manifold에 의해서 support 받는 분포를 학습할 때 cost function이 아니라는 사실을 강조한다.   
    하지만, EM distance는 이러한 환경에서 민감하다.     

------------------------------------------------------------------------------------------------

## Wasserstein GAN       

  + Theorem 2는 최적화가 이루어질 때 Wasserstein distance가 JS divergence에 비해서 더 좋은 특성을 가진다는 사실을 가리킨다.  

  + 하지만, EM distance의 하한 중 가장 큰 값(supremum)은 계산하기 매우 어렵다.     
    반면에, kantorovich-Rubinstein duality는 다음을 설명한다.     

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/6856f15f-c3c2-42c0-915d-73f3c740a890' >
<p>    
   
  + 그리고 이 식을 우리가 만약 어떤 $K$ 에 대해서 $K$ - Lipschitz를 만족하는 parameterized family of functions을 가지고 있다면 다음과 같이 변경할 수 있다고 한다.    

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/36c30fcc-a736-4d81-b858-7325bb50982d' >
<p>    
    
  + 이 문제에 대한 solution $f : x \rightarrow \mathbb{R}$ 을 다음과 같이 구할 수 있다.   

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/3bbb4483-5fb8-42c1-8b7c-0e085897c451' >
<p>    
     
  + 이제 equation (2)에 있던 최대화 문제를 해결하는 함수 $f$ 를 찾는 문제에 도달한다.    

  + 이를 대략 근사하기 위해서, 우리가 할 수 있는 것은 compact space $W$에 놓여있는 가중치 $w$ 를 파라미터로 가지는 neural network를 학습시키는 것으로,    
    그리고 나서 $\mathbb{E}$<sub>$z \sim p(z)$</sub> ${\left \[ \bigtriangledown_\theta {f_w}\cdot g_\theta \left \( z \right \) \right \]}$ 을 통해서 backprop을 한다. 이 과정은 일반적인 GAN을 가지고 수행한다.   

  + $W$이 compact라는 사실은 모든 function $f_w$가 개별 가중치가 아닌 $W$에만 의존하는 일부 $K$에 대한 $K$-Lipschitz일 것 같이라고 암시하므로 (2)를 관련없는 scaling factor와 critic $f_w$의 용량까지 근사한다.    

  + Parameters $w$가 compact space에 놓이도록 하기 위해서(어떤 경계 안에 들어오게 하기 위해서), 우리가 할 수 있는 간단한 무언가는 각 gradient update 이후에 가중치들을 fixed box (예를 들어, $W = \left \[-0.01, 0.01 \right \]^{l}$ 에 고정시키는 것이다.    

  + Weight clipping은 Lipschitz constraint를 강제하기 위해서 명백하게 끔찍한 방법이다.   
    > 만약 clipping parameter가 크면, 어떤 가중치는 이들의 limit까지 도달하는데에 오랜 시간이 걸릴 수 있으며, critic을 최적점까지 학습시키기 어렵게 만든다.   
    > 만약 clipping parameter가 작으면, 이는 layer 수가 적을 때 혹은 batch normalization이 사용되지 않았을 때 vanishing gradient로 이끌기 쉬울 수 있다. 

  + 저자는 거의 차이가 없는 간단한 변형을 가지고 실험했으며, 단순성과 이미 좋은 성능을 내는 것 때문에 weight clipping을 고수하기로 했다.
    > 대신, 신경망 환경에서 Lipschitz constraints를 강제하는 주제를 추가적인 future work로 남겼다.   

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/0283b654-de3b-459a-a6df-ef7bace1ef86' >
<p>    
     
  + EM distance가 연속적이고 미분 가능하다는 사실은 우리가 critic을 최적성까지 훈련시킬 수 있다는 사실을 의미한다.

  + 즉, 우리가 더 많이 critic을 학습시킬수록 더 신뢰할 만한 Wasserstein의 gradient를 얻게 되며, 이는 **Wasserstein의 거의 모든 곳에서 미분 가능하다는 사실**에 의해 실제로 유용하다.

  + JS의 경우에, discriminator가 더 나은 gradients을 얻을수록 더 신뢰할 만한 gradient을 얻게 되지만, JS가 locally saturated이기 때문에 true gradient는 0이고 **vanishing gradient을 얻게 된다**.

  + Figure 2에서 이 개념의 증명을 볼 수 있으며, 여기서 저자는 GAN discriminator와 WGAN critic을 최적성까지 학습하였다.   

<p align='center'>
    <img src='https://github.com/WestChaeVI/GAN/assets/104747868/03af3fc2-93ab-447a-a19e-74d218d012d4' >
<p>  
    
  + Discriminator는 fake와 real 사이를 구별하기 위해 아주 빠르게 학습하며, 예상되듯이 신뢰할 만한 gradient information을 제공하지 못한다.
    > 기울기 거의 무한대이거나 0에 가깝다.   

  + 하지만 critic은 **saturate** 되지 않으며 모든 곳에서 현저하게 명백한 gradient를 제공하는 **linear function**으로 수렴한다.

  + 가중치를 제한한다는 사실은 공간의 서로 다른 부분에서 함수의 가능한 성장을 최대 선형으로 제한하며, 이는 최적의 critic이 이러한 행동을 하도록 강제한다.

  + 더 중요한 점은, critic을 최적점까지 학습할 수 있다는 사실은 학습할 때 modes가 붕괴(collapse)되는 것을 불가능하게 만든다.

------------------------------------------------------------------------------------------------     

## Empirical Results    

+ 저자는 Wasserstein GAN algorithm을 사용하여 image generation에 대해서 실험을 수행하였으며, 표준적인 GAN에서 사용되는 formulation에 비해 WGAN을 사용하는 것이 상당한 이점을 가진다고 주장함.    
  - Generator 수렴과 샘플 품질과 상관관계가 있는 의미 있는 loss metric
  - 최적화 프로세스의 향상된 안정성    

------------------------------------------------------------------------------------------------     

### Meaningful loss metric   

+ WGAN algorithm은 각 generator update (Algorithm 1에서의 10번째 줄) 이전에 critic $f$를 상대적으로 잘 학습하도록 시도하기 때문에, 이 지점에서의 loss function은 저자가 $f$의 Lipschitz constant를 제약하는 방법과 관련되어 있는 constant factor까지의 EM distance의 추정치이다.  

+ 첫 번째 실험은 얼마나 이 추정치가 생성된 sample의 quality와 correlation이 있는지를 보여준다.   

+ Convolution DCGAN architecture 외에, generator 혹은 generator와 critic 모두를 4-layer ReLU-MLP with 512 hidden units로 교체하여 실험 진행

+ Figure 3,4는 모든 3개의 architecture의 WGAN training 동안의 EM distance, JS divergence 의 WGAN estimate의 변화를 보여준다.       

<table style="margin-left: auto; margin-right: auto;">
  <th>
    <p align='center'> Figure 3 (EM estimate)</p>
  </th>
  <th>
    <p align='center'> Figure 4 (JS estimate)</p>
  </th>
  <tr>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/e57a58d4-71e5-47bd-9850-139ee70fca74' width=400, height=400>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/a96459c6-f354-4018-9d94-697de7c20b19'width=400, height=400>
      <p>
  <tr>
    <td>
      <p align='center'>Figure 3는 생성된 sample의 visual quality와 이들의 그래프가 상관관계가 있음을 명확하게 보여준다.<p>
    </td>
    <td>
      <p align='center'>Figure 4는 명백하게 sample의 quality와 상관관계가 없음을 보여준다.<p>
    </td>
  </tr>
  </tr>
</table>    

+ 마지막으로, 높은 learning rate를 사용하거나 critic에 Adam과 같은 momentum based optimizer를 사용할 때 WGAN 학습이 불안정해질 수 있다.   
+ Critic에 사용되는 loss nonstationary 이므로, momentum 기반의 방법론들은 더 안 좋게 수행하는 것으로 보인다.   
+ 저자는 momentum이 잠재적인 원인이라고 파악했는데, 이는 loss가 상승하고 샘플이 약화될 때 Adam step과 gradient 간의 cosine 값이 일반적으로 마이너스로 변하기 때문이라고 한다.   
+ 이 cosine 값이 마이너스인 유일한 곳은 이러한 불안정한 상황들이었다.  
+ 그러므로 저자는 nonstationary problem에서도 잘 작동하는 것으로 알려져 있는 RMSProp을 사용하였다.

------------------------------------------------------------------------------------------------    

### Improved Stability       

+ WGAN의 이점 중 하나는 critic을 optimality까지 학습하도록 해준다는 것이다.

+ critic이 완전 학습되었을 때, 이는 다른 신경만인 generator가 학습될 수 있도록 하는 loss를 제공한다.

+ 이는 우리가 실험할 때 더 이상 generator와 discriminator의 capacity를 적절하게 균형 맞출 필요가 없다는 것을 의미

+ critic이 더 잘할수록, generator가 학습하는 데에 사용되는 더 높은 gradient를 제공한다.     


+ 저자는 generator에 사용되는 구조를 변화시켰을 때 GAN보다도 WGAN이 더 강건하다는 것을 확인했다.  
  - 3개의 generator 구조를 실험해서 확인함.

<table style="margin-left: auto; margin-right: auto;">
  <th>
    <p align='center'> Figure 5 </p>
  </th>
  <th>
    <p align='center'> Figure 6 </p>
  </th>
  <th>
    <p align='center'> Figure 7 </p>
  </th>
  <tr>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/47213317-739c-4538-a6a0-7d97797296b5'>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/6be4c688-faa0-4bf3-8945-467fda1b9cf9'>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/bd61cdac-68c7-46a3-ac28-ad9a088f73a5'>
      <p>
    </td>
  <tr>
    <td>
      <p align='center'>Convolutional DCGAN generator<p>
    </td>
    <td>
      <p align='center'>Convolutional DCGAN generator without batch normalization with 512 hidden units<p>
    </td>
    <td>
      <p align='center'>4-layer ReLU-MLP with 512 hidden units<p>
    </td>
  </tr>
  </tr>
</table>     

+ (2)과 (3)의 경우 GAN을 이용했을 때 매우 안 좋은 성능을 내는 것으로 알려져 있다.  

+ 그래서 저자는 WGAN critic 이나 GAN discriminator로 convolutional DCGAN 구조를 사용했다.   

------------------------------------------------------------------------------------------------      

## Conclusion & Future Work(limitation)      

+ 새로운 모델에서, 학습의 안정성이 향상될 수 있고 mode collapse와 같은 문제들이 사라졌고, debugging이나 hyper-parameter search에 유용하고 유의미한 학습 곡선을 제공한다는 것을 보였다.

+ 더불어서, 최적화 문제가 건전함을 보였으며, 분포 간의 다른 거리와의 깊은 연관성을 강조하는 광범위한 이론적인 논의도 제공되었다.

------------------------------------------------------------------------------------------------      

### 개인적으로 느낀 점   
     
+ technical한 내용보다 어떻게보면 수학적인 내용들로 논문을 구성하고 있는지라..    
  솔직히 말하자면, 논문을 완벽하게 이해했다고 한다면 거짓말이다. 추후에 **반복적으로 읽어야할 논문**이다!

+ 모델을 사용하는 입장에서 기억해야 할 점이라고 한다면, 먼저



