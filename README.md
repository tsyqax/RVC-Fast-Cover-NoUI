# RVC-Fast-Cover-NoUI

<details>
<summary>한국어</summary>
  
아직 완성되지 않은 버전을 사용하고 싶다면 `canary` 브랜치를,  
`canary`보다 안정적인 테스트 버전을 사용하고 싶다면 `last-test-before-release` 브랜치를 사용하세요.  

무료 코랩에서 사용하기 위한 NoUI 환경으로 구성되어 있습니다.  
또한, 기능은 약간 축소되었으나 속도 향상에 초점을 맞추었습니다.  

*⚠️ 주의: 이 환경은 구글 코랩에 맞게 튜닝되고 테스트되었습니다.  
그 외 환경에서 실행하려면 약간의 수정이 필요할 수 있습니다.*

---

## 사용법
### 구글 코랩 링크
**[[한국어]](https://colab.research.google.com/drive/10iTH1SGxQK2TCDfzUpgke1UFBUJHGCnk)** **[[EN]](https://colab.research.google.com/drive/1ki84JkAFXUDIDmj2YHWRX52nhuJ5VOVO)**  

*⚠️ 주의: 유튜브 모드를 사용할 때, 코랩 환경은 일시적인 속도 제한이나 차단 조치에 매우 취약합니다.  
소스 자산은 로컬 파일 업로드나 구글 드라이브 경로를 통해 직접 저장하는 것을 권장합니다.*

### 필요한 것들
- `pip install -r requirements.txt`: 개인적으로는 uv를 사용하는 것을 권장합니다.
- `ffmpeg`: 오디오 처리에 필요
- 이 저장소 복제(`git clone...`)

#### 모델 파일 위치
모델 파일은 `주요 경로/models/모델명/모델파일들`로 위치해야 합니다.  
예: `DIR/models/myModel/model.pth`, `DIR/models/myModel/idx.index`   

## main.py
```bash
python main.py --input "추론대상" --rvc-name "모델명" [ADDITIONAL_ARGUMENTS]
```

### CLI 인자값 참조 (`main.py`)

| 인자값 | 약어 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `-in` | `str` | *필수* | 오디오 입력 소스 (URL이나 경로) |
| `--rvc-name` | `-rvc` | `str` | *필수* | 추론시킬 RVC 모델 이름 (폴더명) |
| `--pitch-vocal` | `-p1` | `float` | `0` | 보컬 트랙에만 적용되는 피치 조정값 (단위: sgs) |
| `--pitch-other` | `-p2` | `float` | `0` | 배경 반주 트랙에만 적용되는 피치 조정값 (단위: sgs) |
| `--sep-mode` | `-sep` | `bool` | `True` | False인 경우 음원 분리 스킵|
| `--index-rate` | `-irate`| `float` | `0.75` | 추론 시 인덱스 비율 |
| `--rms-rate` | `-rms` | `float` | `0.8` | 원본의 소리를 얼마나 따라갈 것인가 |
| `--rvc-method` | `-algo` | `str` | `'rmvpe'` | 피치 추출 알고리즘 (`'rmvpe'` 또는 `'fcpe'`). |
| `--vocal-sound` | `-s1` | `int` | `100` | 보컬 소리 크기 |
| `--other-sound` | `-s2` | `int` | `80` | 배경 반주 소리 크기 |
| `--chorus-mode` | `-chr` | `[0, 1, 2, 3]` | `0` | 코러스 분리 모드 |

* 10 sgs = 1 Octave = 12 Semiton
* sgs는 삼겹살의 약자입니다

#### 코러스 분리 모드
0: 분리하지 않음 (메인 보컬로 추론)  
1: 분리 후 병합 없음 (사라짐)  
2: 분리 후 병합 (반주로 병합)  
3: 분리 후 추론, 이후 병합 (보컬로 병합)  

---
### Thanks to [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen)

</details>

If you want to use the unreleased bleeding-edge version, switch to the `canary` branch.  
For a more stable preview build before the official release, please use the `last-test-before-release` branch.

It is configured with a NoUI environment optimized for use in the free Google Colab.  
While some non-essential features have been slightly streamlined, the core focus has been strictly placed on maximizing execution speed.

*⚠️ Warning: This environment has been extensively tuned and tested specifically for Google Colab. Run configurations on other native OS environments may require minor modifications.*

---

## How to Use
### Google Colab Notebooks
You can run the entire pipeline directly via these links: **[[한국어]](https://colab.research.google.com/drive/10iTH1SGxQK2TCDfzUpgke1UFBUJHGCnk)** **[[EN]](https://colab.research.google.com/drive/1ki84JkAFXUDIDmj2YHWRX52nhuJ5VOVO)**  

*⚠️ Warning: When using YouTube Mode, the Colab environment is highly susceptible to temporary IP rate limits or blocklists. Storing source assets directly via local file uploads or Google Drive paths is strongly recommended.*

### Prerequisites & Installation
- `pip install -r requirements.txt` (Personally, using **`uv`** is highly recommended for blazing-fast setups.)
- `ffmpeg`: Required for core audio processing.
- Clone this repository (`git clone ...`)

#### Model File Directory Structure
Model files must be placed under the specific path format: `ROOT_DIR/models/YOUR_MODEL_NAME/`  
Example: `DIR/models/myModel/model.pth` and `DIR/models/myModel/idx.index`   

---

## main.py Execution
```bash
python main.py --input "SOURCE_PATH_OR_URL" --rvc-name "MODEL_NAME" [ADDITIONAL_ARGUMENTS]
```

### CLI Arguments Reference (`main.py`)

| Argument | Shorthand | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `-in` | `str` | *Required* | Audio input source (URL or local file path) |
| `--rvc-name` | `-rvc` | `str` | *Required* | RVC model name for inference (Target folder name) |
| `--pitch-vocal` | `-p1` | `float` | `0` | Pitch adjustment applied exclusively to the vocal track (Unit: sgs) |
| `--pitch-other` | `-p2` | `float` | `0` | Pitch adjustment applied exclusively to the background instrumental track (Unit: sgs) |
| `--sep-mode` | `-sep` | `bool` | `True` | Bypasses the audio separation step if set to False |
| `--index-rate` | `-irate`| `float` | `0.75` | Index feature multiplier ratio used during inference |
| `--rms-rate` | `-rms` | `float` | `0.8` | Determines how closely the output matches the original volume envelope |
| `--rvc-method` | `-algo` | `str` | `'rmvpe'` | Core pitch extraction method algorithm (`'rmvpe'` or `'fcpe'`) |
| `--vocal-sound` | `-s1` | `int` | `100` | Output volume level for the processed vocal track |
| `--other-sound` | `-s2` | `int` | `80` | Output volume level for the background instrumental track |
| `--chorus-mode` | `-chr` | `[0, 1, 2, 3]` | `0` | Chorus separation mode routing logic |

* **10 sgs = 1 Octave = 12 Semitones**
* *`sgs` is an abbreviation for "Samgyeopsal" (Korean grilled pork belly).*

---
### Thanks to [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen)
