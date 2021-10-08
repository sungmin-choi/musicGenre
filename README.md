# musicGenre 딥러닝을 통한 음악 분류 시스템 구축 및 음악수집을 위한 웹사이트 제작

# 실험환경

**운영체제**:Windows 

- pycham 
- python3 
- vanila JS 
- tensorflow 
- librosa 
- flask 
- dlib
# 구현방법
First, install libs
    pip install opencv-contrib-python==3.4.0.14
    pip install dlib
    pip install tensorflow
    pip install librosa

실행:python main.py ->index.html 열고 로컬호스트에서 사용가능

# 데이터

데이터는 10가지 음악장르 를 분류하고 80%는 데이터 학습에 사용하였고 20%는 테스트 데이터로 사용하였다

![캡처1](https://user-images.githubusercontent.com/53036141/136496927-1254e43b-d3fe-4f47-8ebc-9ac3b1edd3b2.PNG)

# 학습 사용한 모델

기존 CNN에서 정확도를 향상시키기 위해서는 먼저 아키텍처를 구축해야 하며, 컨볼루션 채널, 깊이, 이미지 해상도를 높여 계산량을 늘리고 정확도를 높인다. 그러나 채널 수(너비)/깊이(깊이)/이미지 크기(해상도)를 높이는 방법에 대한 연구가 없기 때문에 효과적인 스케일링 방법을 소개합니다. 또한 Neural Architecture Search를 사용하여 베이스라인 네트워크를 구축하고 스케일링을 수행하며 기존 ConvNet보다 높은 정확도와 효율성으로 Efficient Nets라는 제품 그룹을 달성할 것을 제안합니다. Efficient Net-B7은 기존 ConvNet보다 8.4배 작고 6.1배 빠른 84.3%로 Image Net의 첫 번째 정확도를 달성했습니다.

## 성능 개선을 위해서 Batch Normalization 사용하였다.
최근 널리 사용되고 있는 다양한 고성능 아키텍처 중 BN(Batch Normalization)은 필수 불가결한 요소입니다. BN은 매우 좋은 성능을 발휘할 수 있으므로 BN을 자주 사용하면 다음과 같은 잘 알려진 장점이 있습니다. BN의 가장 큰 장점이라고 하면 학습 속도를 높일 수 있습니다.

# lask 기반 ML, DL RESTFUL API 웹사이트 구축
![캡처2](https://user-images.githubusercontent.com/53036141/136498098-6bb91cbe-701d-4fe0-bc10-c65aa6bd3bbe.PNG)
