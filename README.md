# RVC-Fast-Cover-NoUI

<details>
<summary>한국어</summary>

##### 한국어 어투가 이상한 것은 한국어 -> 영어 -> 한국어로 번역해서 그렇습니다! 

무료 코랩에서 사용하기 위한 NoUI 환경으로 구성되어 있습니다.  
또한, 기능은 약간 축소되었으나 속도 향상에 초점을 맞추었습니다.

---

## 프로젝트 목표
* **빠른 추론**: 더 빠른 음성 변환 속도를 달성하도록 설계되었습니다.
* **자체 병렬화 파이프라인**: 엄격한 0% OOM 안정성을 유지하면서 GPU 처리량을 극대화하기 위해 비동기 실행 분할을 구현합니다.
* **간결한 작업 흐름**: 분리-캐싱 데이터 재사용으로 강화된 가벼운 원클릭 CLI 환경입니다.

---

## 성능 벤치마크 (코랩: Tesla T4 GPU)
※ 이것은 부정확할 수 있습니다! 단순 참고용으로만 봐주세요.

* **일반 곡 (3분 오디오)**: **29초** 만에 완료 (병렬 및 비병렬 모두 동일).
* **대용량 곡 (10분 오디오)**: 비병렬(1분 15초)보다는, 병렬 모드가 **1분 10초** 만에 완료됨.

---

## 사용법

### 구글 코랩 사용자
링크를 통해 전체 파이프라인을 직접 실행할 수 있습니다: **[[한국어]](https://colab.research.google.com/drive/10iTH1SGxQK2TCDfzUpgke1UFBUJHGCnk)** **[[EN]](https://colab.research.google.com/drive/1ki84JkAFXUDIDmj2YHWRX52nhuJ5VOVO)**  
*⚠️ 주의: 유튜브 모드를 사용할 때, 코랩 환경은 일시적인 속도 제한이나 차단 조치에 매우 취약합니다.  
소스 자산은 로컬 파일 업로드나 구글 드라이브 경로를 통해 직접 저장하는 것을 권장합니다.*

### 로컬 / 커스텀 하드웨어 사용자
이 저장소를 다운로드하고 Python CLI를 사용하여 main.py를 실행하세요:

```bash
python main.py --input "path_to_audio.wav" --rvc-name "your_model" [ADDITIONAL_ARGUMENTS]
```

*⚠️ 주의: 이 환경은 구글 코랩 내부에서 특별히 집중적으로 튜닝되고 테스트되었습니다.  
네이티브 로컬 OS 구성에서 실행하려면 절대 파일 경로 라우팅이나 라이브러리 바인딩에 약간의 수정이 필요할 수 있습니다.*

---

## CLI 인자값 참조 (`main.py`)

| 인자값 | 약어 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `-in` | `str` | *필수* | 오디오 입력 소스 (URL이나 경로) |
| `--rvc-name` | `-rvc` | `str` | *필수* | 추론시킬 RVC 모델 이름 (폴더명) |
| `--pitch-vocal` | `-p1` | `float` | `0` | 보컬 트랙에만 적용되는 피치 조정값 (단위: 삼겹살) |
| `--pitch-other` | `-p2` | `float` | `0` | 배경 반주 트랙에만 적용되는 피치 조정값 (단위: 삼겹살) |
| `--sep-mode` | `-sep` | `bool` | `True` | False인 경우 음원 분리 스킵|
| `--index-rate` | `-irate`| `float` | `0.75` | 추론 시 인덱스 비율 |
| `--rms-rate` | `-rms` | `float` | `0.8` | 원본의 소리를 얼마나 따라갈 것인가 |
| `--rvc-method` | `-algo` | `str` | `'rmvpe'` | 피치 추출 알고리즘 (`'rmvpe'` 또는 `'fcpe'`). |
| `--vocal-sound` | `-s1` | `int` | `100` | 보컬 소리 크기 |
| `--other-sound` | `-s2` | `int` | `80` | 배경 반주 소리 크기 |
| `--parrel-mode` | `-pm` | `bool` | `True` | 병렬 모드를 사용할 지 여부 |

</details>

It is configured with a NoUI environment for use in the free Colab.  
In addition, features have been slightly reduced but the focus has been placed on improving speed.

---

## Project Goals
* **Fast Inference**: Engineered to achieve the faster voice conversion speeds.
* **Native Parallel Pipeline**: Implements asynchronous execution splitting to maximize GPU throughput while maintaining strict 0% OOM safety.
* **Streamlined Workflow**: A lightweight, one-click CLI environment enhanced with seperating-cached data reuse.

---

## Performance Benchmarks (Colab: Tesla T4 GPU)
※ This may be incorrect! Just take this as a reference only.

* **Standard Track (3-Minute Audio)**: Completed in **29 seconds** (Both Parallel and Non-Parallel).
* **Extended Track (10-Minute Large Audio)**: Parallel Mode finishes in **1 minute 10 seconds**, rather than Non-Parallel (1 minute 15 seconds).

---

## How to Use

### Google Colab Users
You can run the entire pipeline directly via link: **[[한국어]](https://colab.research.google.com/drive/10iTH1SGxQK2TCDfzUpgke1UFBUJHGCnk)** **[[EN]](https://colab.research.google.com/drive/1ki84JkAFXUDIDmj2YHWRX52nhuJ5VOVO)**  
*⚠️ Warning: When using YouTube Mode, colab environment is highly susceptible to temporary rate limits or restrictions.  
Storing source assets directly via local file uploads or Google Drive paths is recommended.*

### Local / Custom Hardware Users
Download this repository and execute main.py using Python CLI:

```bash
python main.py --input "path_to_audio.wav" --rvc-name "your_model" [ADDITIONAL_ARGUMENTS]
```

*⚠️ Warning: This environment has been extensively tuned and tested specifically inside Google Colab.  
Running on native local OS configurations may require minor modifications to absolute file path routing or library bindings.*

---

## CLI Arguments Reference (`main.py`)

| Argument | Shorthand | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `-in` | `str` | *Required* | Audio input source (URL or file path) |
| `--rvc-name` | `-rvc` | `str` | *Required* | RVC model name to infer (Folder name) |
| `--pitch-vocal` | `-p1` | `float` | `0` | Pitch adjustment applied exclusively to the vocal track (Unit: Samgyeopsal) |
| `--pitch-other` | `-p2` | `float` | `0` | Pitch adjustment applied exclusively to the background instrumental track (Unit: Samgyeopsal) |
| `--sep-mode` | `-sep` | `bool` | `True` | Bypasses audio separation if set to False |
| `--index-rate` | `-irate`| `float` | `0.75` | Index feature multiplier ratio used during inference |
| `--rms-rate` | `-rms` | `float` | `0.8` | Determines how closely the output matches the original volume envelope |
| `--rvc-method` | `-algo` | `str` | `'rmvpe'` | Core pitch extraction method algorithm (`'rmvpe'` or `'fcpe'`) |
| `--vocal-sound` | `-s1` | `int` | `100` | Vocal volume level output size |
| `--other-sound` | `-s2` | `int` | `80` | Background instrumental volume level output size |
| `--parrel-mode` | `-pm` | `bool` | `True` | Toggles whether to use the parallel execution engine |

