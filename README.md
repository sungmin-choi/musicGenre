# musicGenre 딥러닝을 통한 음악 분류 시스템 구축 및 음악수집을 위한 웹사이트 제작

# 실험환경

**운영체제**:Windows  
**서버구축**:- flask  
**머신러닝 구축**: tensorflow  
**프론트엔드 언어**:vanila JS  
**백엔드 서버 구축 언어**:python3   
**IDE**:pycham    
**라이브러리**:librosa , dlib 

# 구현방법
First, install libs
    pip install opencv-contrib-python==3.4.0.14
    pip install dlib
    pip install tensorflow
    pip install librosa

실행:python main.py ->index.html 열고 로컬호스트에서 사용가능(127.0.0.1:8000 으로 로컬서버에서 웹사이트 접속 가능)

# 데이터

데이터는 10가지 음악장르 를 분류하고 80%는 데이터 학습에 사용하였고 20%는 테스트 데이터로 사용하였다

![캡처1](https://user-images.githubusercontent.com/53036141/136496927-1254e43b-d3fe-4f47-8ebc-9ac3b1edd3b2.PNG)

# 학습 사용한 모델

기존 CNN에서 정확도를 향상시키기 위해서는 먼저 아키텍처를 구축해야 하며, 컨볼루션 채널, 깊이, 이미지 해상도를 높여 계산량을 늘리고 정확도를 높인다. 그러나 채널 수(너비)/깊이(깊이)/이미지 크기(해상도)를 높이는 방법에 대한 연구가 없기 때문에 효과적인 스케일링 방법을 소개합니다. 또한 Neural Architecture Search를 사용하여 베이스라인 네트워크를 구축하고 스케일링을 수행하며 기존 ConvNet보다 높은 정확도와 효율성으로 Efficient Nets라는 제품 그룹을 달성할 것을 제안합니다. Efficient Net-B7은 기존 ConvNet보다 8.4배 작고 6.1배 빠른 84.3%로 Image Net의 첫 번째 정확도를 달성했습니다.

## 성능 개선을 위해서 Batch Normalization 사용하였다.
최근 널리 사용되고 있는 다양한 고성능 아키텍처 중 BN(Batch Normalization)은 필수 불가결한 요소입니다. BN은 매우 좋은 성능을 발휘할 수 있으므로 BN을 자주 사용하면 다음과 같은 잘 알려진 장점이 있습니다. BN의 가장 큰 장점이라고 하면 학습 속도를 높일 수 있습니다.

# flask 기반 ML, DL RESTFUL API 웹사이트 구축
![캡처2](https://user-images.githubusercontent.com/53036141/136498098-6bb91cbe-701d-4fe0-bc10-c65aa6bd3bbe.PNG)서
서버를 사용해서 딥러닝 모델 데이터 전처리 결과값을 제공하는데 사용한다,flask는 매우 가벼운 프레임워크로 최근에는 python 의 강력한 기계 학습 관련 기능 패키지를 통해 머신러닝 및 딥러닝 결과에서 주로 추출한 모델을 빠르게 개발할 수 있습니다.

# 메인 페이지
![캡처3](https://user-images.githubusercontent.com/53036141/136500346-067a47c9-fb9f-4805-aa75-4e1af4a596f3.PNG)
1.녹음기능버튼(route /record로 이동) 노래를 녹음가능한 페이지로 이동한다.녹음 완료후 wav파일로 저장한다.  
2.업로드 버튼 (wav파일 업로드 기능)  
3.예측버튼(파랑색) 누르면 음악장르를 예측한 결과가 나옵니다.  
4.데이터 증강 버튼 wav파일 추가를해서 모델에 데이터를 증가시켜서 다시 재학습을 통해서 더 정교한 모델을 만들기 위함이다.  
5.증강데이터 와 장르를 선택해서 추가를 해줘야합니다   
6.증강 데이터 확인 버튼(갈색) 누르면 모델에 데이터가 저장된다.  
7.재학습 버튼(주황색) 누르면 모델 재학습을 도와줍니다 (*단점* 시간이 너무 오래 걸린다)  

# 녹음 후 wav파일로 저장하는 과정
![image](https://user-images.githubusercontent.com/53036141/136501243-38f8093d-bbdf-4038-a529-06468a5896c8.png)

# 결과 예측
![image](https://user-images.githubusercontent.com/53036141/136501300-d2f2c1da-e074-42a0-852c-b6417ef607e5.png)


# 데이터 업로드 과정 
![image](https://user-images.githubusercontent.com/53036141/136501388-31c38e4e-2818-40bd-8b44-5a4343105a73.png)

# 예측결과 성능도
![image](https://user-images.githubusercontent.com/53036141/136501500-eef4ceb1-4f2b-4f61-b8b4-0849932e4155.png)
주황색 선은 검증 데이터 세트의 훈련 결과 그래프이고 파란색 선은 훈련 데이터 세트의 훈련 결과 그래프입니다. 정확도의 경우 훈련 데이터 세트의 정확도가 크게 다르지 않음을 알 수 있으며 모두 90% 이상으로 좋은 성능을 보입니다.loss 값도 준수한 값을 가지고 있는걸 알 수 있습니다.
