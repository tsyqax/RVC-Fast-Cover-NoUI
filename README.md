### **RVC-FAST-COVER-NOUI**

## 현재 제작 중  
(오류 및 버그 탐색 중, 작동 보장 못함)

### 특징  

#### 코랩 기준 제작 (UI 없음)  
코랩을 기준으로 제작되었습니다.  
로컬 환경 또는  윈도우 환경에서는 수정이 필요할 수 있습니다.

**[코랩 링크](https://colab.research.google.com/drive/10iTH1SGxQK2TCDfzUpgke1UFBUJHGCnk)**  

#### 빠른 수행 (예상)  

1. **일부 과정 생략**  
De-Reverb (리버브 제거) 등의 생략이 대표적입니다.  
후처리 리버브도 추가하지 않았습니다.

3. **uv 사용 및 패키지 간소화...랄까?**  <br>
빠른 설치를 위해 UV를 사용했습니다.  
패키지 목록도 일부 줄었습니다.

5. **병렬 연산 구현 (베타)**    
RVC 내부 추론에서 간단한 병렬 연산을 구현했습니다.  
1분 이상의 오디오는 코어 수대로 분할하여 병렬로 처리합니다.  
리소스는 좀 더 먹겠지만, 속도는 늘어날 것으로 예상하고 있습니다.  

4. **rmvpe 및 fcpe 탑재**  
그 외 f0 모델은 버렸습니다.  
사실 제가 쓰고자 하는 것으로 만든거라,  
rmvpe를 위주로 하고, 새로 나온 것처럼 보이는 fcpe를 추가했습니다.   

#### 기타 도움  
음원 분리는 demucs를 통해 이루어집니다.  
Github: SociallyIneptWeeb/AICoverGen 및 Colab: CoverGen_NO_UI_v2 에서 많은 아이디어를 얻었습니다.  
코랩과 Gemini 와 함께 개발했습니다.

---

## Currently in Development  
(Searching for errors and bugs; operation is not guaranteed)

### Features  

#### Created for Colab (for free users)  
This was made for Colab.  
Modifications may be necessary for local and Windows environments.

**[COLAB LINK](https://colab.research.google.com/drive/1ki84JkAFXUDIDmj2YHWRX52nhuJ5VOVO)**   

#### Fast Performance (Expected)  

1. **Some processes omitted**  
Processes like De-Reverb have been omitted.  
Post-processing reverb has not been added either.

2. **Uses uv and simplified packages**  
uv was used for faster installation.  
The list of packages has also been reduced.

3. **Parallel processing implemented**  
Simple parallel processing has been implemented for RVC internal inference.  
Audio files longer than one minute are split and processed in parallel according to the number of cores.  
It's expected to consume more resources, but the speed should increase.

5. **Includes rmvpe and fcpe**  
Other f0 models were discarded.  
This was made for my own use, so I focused on rmvpe and added the newly released-looking fcpe.

#### Other Information  
Music source separation is done through demucs.  
I got many ideas from Github: SociallyIneptWeeb/AICoverGen and Colab: CoverGen_NO_UI_v2.  
This project was developed with Colab and Gemini.
