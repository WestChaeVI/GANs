### Reference : https://github.com/YoojLee/paper_review/tree/main/code/cycle_gan    

# 구현 시 주의 사항     

GAN을 학습시킬 때에 가장 중요한 점은 Generator와 Discriminator의 alternative training을 잘 제어해줘야 하는 것이다.    
G의 gradient를 D의 업데이트 과정에서 차단을 하고, 그 반대도 역시 G의 업데이트 과정에서 D의 gradient를 차단해줘야 한다.   

+ Generator forward & backward 과정에서 Discriminator의 gradient를 흐르지 않도록 차단해주는 것이 필요함. (중요)    
  > Generator의 Loss에는 무조건 D가 들어가게 되어 있는데, 이를 Backward하는 과정에서 무조건 discriminator의 gradient가 계산되기 때문      

+ 이와 마찬가지로 Generator의 gradient를 Discriminator 최적화 과정에서 흐르지 않도록 차단해주는 게 필요.
  > 이는 Generator의 output tensor를 계산 그래프에서 **detach**하는 과정을 통해서 수행 가능    
  
------------------------------------------------------------------------------------------------------------------------      

# 실험 결과   

## 1. Loss 양상   

<table style="margin-left: auto; margin-right: auto;">

  <th style="text-align:center"> </th>
  <th>
    <p align='center'>Gerneratore</p>
  </th>
  <th>
    <p align='center'>Discriminator</p>
  </th>
  <tr>
    <td style="text-align:center">Horse to Zebra</td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/2ad9f46c-6063-4acb-8618-a828d5a2a7ad' height=300>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/e20e7898-7cea-4c36-84d4-c3709f02e85d'height=300>
      <p>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">Zebra to Horse</td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/8010a857-06d3-4d69-95f4-c7dbca49d677'height=300>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/ebab1613-4d0e-46cb-ae08-f8f2564e2445'height=300>
      <p>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">Cycle Consistency Loss</td>
    <td colspan='3'>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/262ac638-f3e9-4c91-b0b0-bc5d5bd8a6f5'>
      <p>
    </td>
  </tr>
</table>  


### Cycle Consistency Loss    

Cyc Loss의 경우 Generator와 관련된 loss이기 때문에 Generator 학습 시에 계산하고 Backward 시켜줘야 함. (generator loss 계산 시, cyc loss까지 한꺼번에 계산)    

정상적인 학습 형태인 경우 loss값이 어떤 값을 갖느냐보다 Discriminator와 Generator의 loss 변화 양상을 고려해야 한다는 점을 알 수 있음.  

Discriminator와 Generator가 서로를 견제하며 학습이 되는 상황이기 때문에 어느 하나가 우위를 점하는 것보다 아주 조금씩 변화해가면서 서로 증감 추이가 정반대가 되는 형태가 오히려 학습에 좋은 신호     

------------------------------------------------------------------------------------------------------------------------

# 결과 이미지 시각화   
## Train (epoch : 150 / 200)     

+ 학습 과정 중 변환된 이미지를 저장하여 시각화 (original - translated - reconstructed)


![image](https://github.com/WestChaeVI/GAN/assets/104747868/6c46eebe-2d33-4fde-9fe5-8f84e43252dd)
![image](https://github.com/WestChaeVI/GAN/assets/104747868/5250af5e-9dcc-4044-8f5f-731ff00721b4)
![image](https://github.com/WestChaeVI/GAN/assets/104747868/21cc3a01-221e-49c2-86e3-722ff6dd9fb5)


![image](https://github.com/WestChaeVI/GAN/assets/104747868/6d24624e-0b33-4afa-827f-46cf7da2b261)
![image](https://github.com/WestChaeVI/GAN/assets/104747868/23235b0d-9aa1-45cc-b16c-bef54437cf47)
![image](https://github.com/WestChaeVI/GAN/assets/104747868/e9f38872-cfb5-4814-8925-47cdd34a002f)      

## Test    

<table style="margin-left: auto; margin-right: auto;">

  <thead>
  <th colspan='3'>Orginal - translated - reconstructed</th>
  </thead>
  <tbody>
  <th style="text-align:center"> </th>
  <th>
    <p align='center'>Success case</p>
  </th>
  <th>
    <p align='center'>Failure case</p>
  </th>
  <tr>
    <td style="text-align:center">Horse to Zebra</td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/97a32236-a0c0-4ea8-b6b7-80bd8193c56f' width=400>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/3990d393-a5d3-4055-a5e0-317a6c4c696c'width=400>
      <p>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">Zebra to Horse</td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/3dfc408f-e777-40b7-ada0-06c6e6917f0b'width=400>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/GAN/assets/104747868/5003cf18-abac-49c1-9c8a-28d8c94f48a6'width=400>
      <p>
    </td>
  </tr>
  </tbody>
</table>     


+ 가장 마지막 epoch인 200을 기준으로 testset에 대해서 Image translation을 수행해보고, **비교적** 성공적인 case와 실패한 case를 나눠봄.     

#### Success case (비교적)     

+ Horse to Zebra    
  - 대부분 학습이 잘 되었지만 여러 말들이 겁쳐있었기 때문에 다리 부분들이 몸통 뒷부분(?)으로 변환되었던 것 같다.    
+ Zebra to Horse    
  - 전체 결과 중 제일 잘 나온 이미지라고 생각된다. 얼굴부분의 살짝 아쉬운 것 빼고는 잘 학습된 것 같다.    

#### Failure case        

+ Horse to Zebra    
  - 말의 색깔과 배경의 색깔이 거의 흡사해 학습할 때 방해가 되었던 것 같다. 그래서인지 변환하는 과정에서 배경에서도 얼룩무늬를 띄었던 것 같다.        
+ Zebra to Horse    
  - 몸통이나 다리 같은 부분은 상대적으로 괜찮게 변환되었으나 얼굴 부분의 디테일은 제대로 학습이 되지 않았다.         
    > 이러한 부분은 전처리 과정에서 resized crop을 통해 해결할 수 있을 것 같다.     



























