# 차량 감속기 고장 진단 AI 모델 및 무선 통신 통합 시스템

본 저장소는 철도 및 특수목적 차량 감속기(Gearbox/Reducer)의 실시간 결함 진단을 위한 **2단계 계층형 멀티모달 AI 진단 파이프라인(Multimodal Realtime Inference Pipeline)**과 **저지연 무선 전송 시스템(BLE 5.0 GATT Telemetry)**의 통합 구현체입니다.

---

## 1. 전체 시스템 아키텍처 (System Architecture)

### 1.1. 전체 하드웨어 및 소프트웨어 데이터 흐름도

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                [ Edge Board : Raspberry Pi 5 ]                              │
│                                                                                             │
│  [ main.py : Multi-Threaded Real-time Diagnostics Orchestrator ]                            │
│  │                                                                                          │
│  ├── [Thread 1] Sound Pipeline (sound/sound_main_run.py)                                    │
│  │    ├── 1초 마이크 샘플링 (48,000 Hz Audio PCM)                                           │
│  │    ├── 4차 Butterworth Bandpass Filter (1.8 kHz ~ 10.5 kHz 노면 노이즈 제거)              │
│  │    ├── 24차원 Speed-Invariant 통계 및 DSP 피처 추출 (Hilbert, STFT, Modulation Index)    │
│  │    ├── StandardScaler 정규화 (rpi_scaler_feat.joblib)                                    │
│  │    └── 1단계 XGBoost 고장 스크리닝 분류 (rpi_xgboost_no_gap.joblib) ───┐                  │
│  │                                                                        │ (Trigger)       │
│  └── [Thread 2] CAN Pipeline (SW/rpi_deploy_package/can_inference.py)     ▼                 │
│       ├── SocketCAN 100Hz 수신 버퍼링 (0x386, 0x220, 0x2b0, 0x371)                          │
│       ├── 5개 서브윈도우 (100ms) Zero-Order Hold (ZOH) 시간 동기화 및 41차원 기구학 피처 추출 │
│       ├── 2단계 DualAutoEncoder + CrossModalAttention 잠재 융합 (16D Latent)                │
│       └── Latent FastHead 선형 분류기 / HistGradientBoosting (40차원 통합 잠재 벡터)        │
│                                           │                                                 │
│                                           ▼ (Result: 0 Normal / 1 Abnormal)                 │
│  [ ble_server.py : Zero-Copy IPC & BLE 5.0 GATT Server ]                                    │
│   ├── 1. 20개 서브윈도우 (50ms) DSP 연산 (RMSE 및 FFT Spectral Bandwidth)                   │
│   ├── 2. 164 Byte 바이너리 패킷 Little-Endian 직렬화                                        │
│   ├── 3. POSIX Shared Memory (/ble_shm, 164B) 무복사(Zero-Copy) 기록                       │
│   └── 4. BlueZ D-Bus 기반 GATT Characteristic Notification 브로드캐스트                     │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │ (BLE 5.0 LE GATT Notifications, 100ms Interval)
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                [ Remote Terminal : PC / Client ]                            │
│                                                                                             │
│  [ ble_receiver.py : Bleak Client Telemetry ]                                               │
│   ├── 1. BlueZ AutoAgent 무인 자동 페어링/승인                                              │
│   ├── 2. 타겟 MAC (2C:CF:67:63:37:AA) 탐색 및 GATT 구독                                     │
│   └── 3. 164B 바이너리 패킷 역직렬화 (고장 상태, 20개 서브구간 실효 음압 및 대역폭 실시간 로깅) │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```


---

## 2. AI 진단 모델 상세 명세 (AI Model Specification)

### 2.1. 1단계 음향 기반 이상 감지 모델 (Stage 1 Acoustic Anomaly Screening)
* **모듈 파일**: [sound_main_run.py](sound/sound_main_run.py)
* **목적**: 마이크 음원 신호에서 구동 소음 및 충격파 발생 여부를 1차 고속 필터링하여 이상 의심 시점에 CAN 정밀 융합 엔진을 트리거.
* **전처리 알고리즘**:
  * 4차 Butterworth Bandpass Filter: 통과 대역 $1,800\text{ Hz} \sim 10,500\text{ Hz}$를 설정하여 노면 접지 저주파 노이즈($< 1.8\text{ kHz}$) 및 전자기 고주파 잡음($> 10.5\text{ kHz}$) 차단.
  * 24개 속도 비의존(Speed-Invariant) 음향 및 DSP 피처:
    1. 파형 기본 형상 지표: Crest Factor ($C = \frac{x_{\text{peak}}}{x_{\text{rms}}}$), Shape Factor, Impulse Factor, Margin Factor, Kurtosis, Skewness.
    2. Hilbert Transform 포락선 지표: 포락선 신호 $e(t) = |x(t) + j \mathcal{H}\{x(t)\}|$ 기반 Envelope Crest, Kurtosis, Skewness, 변조 지수 $m = \frac{\sigma_e}{\mu_e}$.
    3. 충격파 특화 피처: 95th Percentile 초과 충격파 에너지 비율($E_{\text{ratio}} = \frac{\sum_{e \ge P_{95}} e^2}{\sum e^2}$), 펄스 카운트.
    4. STFT 주파수 지표: Spectral Flatness, Spectral Entropy, Spectral Rolloff (85%), Spectral Centroid, 2.8kHz 기어 결함 집중 대역비, 고조파 성분비(4~10kHz), 4개 쿼터 서브밴드 파워 비율, 기어 맞물림 주파수(GMF) 추정치.
* **분류 알고리즘**: XGBoost (`rpi_xgboost_no_gap.joblib`) 경량 모델 추론.

### 2.2. 2단계 멀티모달 잠재 융합 모델 (Stage 2 Multimodal Latent Cross-Attention Fusion)
* **모듈 파일**: [can_inference.py](SW/rpi_deploy_package/can_inference.py), [can_models.py](SW/rpi_deploy_package/src/can_models.py)
* **목적**: 1차 음향 이상 감지 시 CAN 기구학 데이터(41차원)와 음향 특징점(24차원) 간 상호 모달리티 정보를 결합하여 감속기 고장 여부를 정밀 판정.
* **전처리 및 특징 추출**:
  * SocketCAN 수신 프레임 4종 해독: 휠속도(`0x386`), 횡가속도/요율(`0x220`), 조향각/속도(`0x2b0`), 페달 조작량(`0x371`).
  * 1초 수집 프레임을 5개의 100ms 서브윈도우로 분할하고 Zero-Order Hold (ZOH) 보간 적용.
  * 41개 차량 동역학/기구학 물리 피처 추출:
    1. 4륜 휠속도 기본 통계량 10종: Mean, Std, Peak-to-Peak, RMS, Variance, Peak, Crest Factor, Shape Factor, Skewness, Kurtosis.
    2. 4륜 간 상대 속도차 및 전후/좌우 편차 8종: $v_{\text{diff, FL-RL}}$, $v_{\text{diff, FR-RR}}$, $v_{\text{diff, FL-FR}}$, $v_{\text{diff, RL-RR}}$ 통계량.
    3. 차체 거동 동역학 4종: 횡가속도(Lateral Acceleration), 요율(Yaw Rate) 평균 및 표준편차.
    4. 조향 시스템 특성 4종: 조향각(Steering Angle) 평균/표준편차/P2P, 조향각속도(Steering Speed) 평균.
    5. 운전자 조작 특성 4종: 가속 페달 개도율 평균/최대치, 제동 페달 조작량 평균/최대치.
    6. 무차원 슬립 비율 7종: $s_{\text{FL}}, s_{\text{FR}}, s_{\text{RL}}, s_{\text{RR}} = \frac{v}{\max(v_{\text{target}}, 1.0)}$, 전후륜 슬립차, 정규화 요율, 정규화 횡가속도, 무차원 표준편차/P2P, 페달-슬립 응답 계수.
* **딥러닝 잠재 융합 신경망 아키텍처**:
  * `GatedResidualBlock`: 입력 벡터를 Linear 레이어, LayerNorm, GELU 활성화 함수, 잔차 연결(Residual Connection) 구조로 인코딩.
  * `DualAutoEncoder`:
    * CAN 인코더: $41 \to 32 \to 16$ 차원 축소
    * Audio 인코더: $24 \to 32 \to 16$ 차원 축소
    * $L_2$ 정규화: $\tilde{z}_c = \frac{z_c}{\|z_c\|_2}$, $\tilde{z}_a = \frac{z_a}{\|z_a\|_2}$
  * `CrossModalAttention`: CAN과 Audio의 상호 Query-Key-Value Scaled Dot-Product Attention 계산:
    $$\text{Attn}_{c \to a} = \sigma\left(\frac{Q_c K_a^T}{\sqrt{d}}\right) V_a, \quad \text{Attn}_{a \to c} = \sigma\left(\frac{Q_a K_c^T}{\sqrt{d}}\right) V_c$$
  * 잠재 공간 기하학 연산:
    * 코사인 유사도: $S_{\text{cos}} = \tilde{z}_c \cdot \tilde{z}_a \in [-1.0, 1.0]$
    * 잠재 유클리드 거리: $D_{\text{latent}} = \sqrt{\max\left(0.0, 2(1 - S_{\text{cos}})\right)}$
  * 통합 40차원 잠재 융합 벡터 구성:
    $$x_{\text{fused}} = \left[ z_{c,\text{att}} \,(16\text{D}) \;\|\; z_{a,\text{att}} \,(16\text{D}) \;\|\; S_{\text{cos}} \,(1\text{D}) \;\|\; D_{\text{latent}} \,(1\text{D}) \;\|\; E_{\text{physics}} \,(6\text{D}) \right] \in \mathbb{R}^{40}$$
  * `FastHead` 선형 로지스틱 분류기:
    $$P(\text{Abnormal}) = \sigma\left(w^T x_{\text{fused}} + b\right)$$
    * 가중치 벡터 $w$와 절편 $b$를 직접 행렬곱 연산하여 마이크로초($\mu\text{s}$) 단위 초고속 에지 추론 수행.

### 2.3. 공학적 자체 검증 및 물리적 한계점 분석 (Critical Engineering Validation)
* **합성 데이터셋 평가 왜곡 주의**:
  * 5-Fold Cross Validation 기준 ROC-AUC 99.97% 및 Accuracy 99.37%를 기록하였으나, 이는 동일 주행 세션 환경의 단일 고장 라벨(냉각수 제거)에 국한된 평가입니다.
* **오디오 마이크 세션 교란(Session Confounder)**:
  * 마이크 설치 위치, 배경 노이즈 지문이 고장 특성으로 오인식되는 지름길 학습(Shortcut Learning)을 방지하기 위해 BPF 필터링 및 대역 정규화를 적용하였으며, 실차 배포 환경에서는 비지도 이상 탐지(Mahalanobis Distance, LOF) 아키텍처와의 앙상블 상호 검증이 요구됩니다.
* **미학습 고장(Zero-Shot Anomaly) 대응**:
  * 실차 환경에서 발생 가능한 기어 마모, 치면 결손 등 새로운 유형의 기계적 고장에 대해서는 지도학습 단독 결정 대신 잠재 유클리드 거리($D_{\text{latent}} \ge 1.0$) 기반의 비지도 거리 판별 로직을 병행 적용합니다.

---

## 3. 실행 방법

### (1) 송신기 실행 

```bash
cd /home/pi/Documents/KRRI_2026
conda activate etrl

# 1. 블루투스 GATT 서버 실행 (별도 터미널 또는 백그라운드)
python ble_server.py

# 2. 메인 AI 진단 실행
python main.py
```

### (2) 수신기 실행 (Remote PC)

```bash
cd /home/team4/Documents/etrl/src
python ble_receiver.py
```

---

## 4. 실시간 출력 화면

### 송신기 콘솔 출력 ([main.py](main.py))
```text
Sound 판단: 0 (Normal)
Sound 판단: 0 (Normal)
Sound 판단: 1 (Abnormal)
CAN 판단: Abnormal
```

### 수신기 콘솔 출력 ([ble_receiver.py](ble_receiver.py))
```text
2026-09-29 11:28:50 Status: NORMAL (0) | dB: 43.1 dB
2026-09-29 11:28:51 Status: NORMAL (0) | dB: 44.3 dB
2026-09-29 11:28:52 Status: ABNORMAL (1) | dB: 52.8 dB
```
