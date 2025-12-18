# 🧠 Fake Text Detection AI

### 파인튜닝을 통한 고성능 가짜 텍스트 분류 모델 개발
---
## 📁 프로젝트 구조

```text
Fake-Text-Detection-AI/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```
---
## 1. 프로젝트 개요 및 목표
온라인 환경에서 가짜·조작된 텍스트(댓글, 리뷰 등)의 증가는  
사용자의 의사결정과 정보 신뢰도를 크게 저하시킨다.  

본 프로젝트는 이러한 문제를 해결하기 위해  
**ELECTRA 기반 딥러닝 모델을 활용한 가짜 텍스트 자동 탐지 시스템**을 구축하는 것을 목표로 한다.

- **프로젝트 유형**: 팀별 미니 프로젝트  
- **문제 유형**: 이진 분류 (Fake vs Real Text)

---

## 2. Task 정의 (Task Definition)
### 🎯 목적
가짜 텍스트를 자동으로 분류하는 **이진 분류 모델 개발**

### 📌 핵심 요구사항
- **분류 문제**: 진짜 텍스트(1) vs 가짜 텍스트(0)
- **목표 성능**: F1-score 85% 이상
- **한국어 처리**: 한국어 댓글·리뷰 특성 반영
- **실시간 추론**: 웹 서비스에서 활용 가능한 추론 속도

---

## 3. Model 선택 (Model Selection)
### 🔹 KcELECTRA (beomi/KcELECTRA-base-v2022)
- Korean comments ELECTRA
- 한국어 댓글, 리뷰, SNS 데이터로 사전학습된 언어 모델
- 비정형·축약 표현이 많은 텍스트에 강점
- BERT 대비 약 30% 빠른 추론 속도

📌 **가짜 텍스트는 문법보다 표현 패턴이 중요하므로  
한국어 댓글에 특화된 KcELECTRA 모델을 선택함**

---

## 4. 기술 스택
- **Language**: Python
- **Framework**: PyTorch (2.x)
- **Libraries**
  - Hugging Face Transformers (4.36.0)
  - Datasets
  - scikit-learn
- **Environment**: Google Colab (GPU)
- **Demo**: Streamlit 웹앱

---

## 5. Data 준비
### 데이터 구성
1. **가짜 댓글 데이터**
   - ChatGPT를 활용하여 가짜 댓글 생성
   - 라벨: `0` (Fake)

2. **진짜 댓글 데이터**
   - 네이버 뉴스 기사 댓글 크롤링
   - 실제 사용자 댓글 활용
   - 라벨: `1` (Real)

3. **데이터 정제 및 통합**
   - CSV 파일로 통합
   - 데이터 구조: `id`, `comment`, `label`

---

## 6. 모델 검증 및 사전 추론 (Validation & Inference)
### 6-1. 모델 검증 (Validation)
- **목적**: 사전학습 모델의 가짜 텍스트 탐지 적합성 확인
- **방법**:  
  전체 데이터셋 중 일부를 검증 데이터로 분리하여 성능 평가

#### 📊 검증 결과
- Accuracy: 45.0%
- Precision: 43.6%
- Recall: 100.0%
- F1-score: 60.7%

📌 **가짜 텍스트를 놓치지 않는 Recall 중심 모델임을 확인**

---

### 6-2. 사전 테스트 추론 (Inference)
- **목적**: 실제 텍스트 입력 시 모델 예측 결과 확인
- **방법**
  - 리뷰/댓글 문장 1개 입력
  - 가짜/진짜 확률 출력

📌 정량 평가(Validation) + 정성 평가(Inference) 병행

---

## 7. Fine-tuning을 위한 환경 세팅
- **개발 환경**: Google Colab (Python 3.x, GPU)
- **사용 라이브러리**: transformers, datasets, torch, scikit-learn, accelerate
- **Tokenizer & Model**: Hugging Face Hub에서 로드
- **데이터 분할**: Train / Validation / Test

### 주요 하이퍼파라미터
- Epoch: 3
- Batch size: 16
- Learning rate: 2e-5
- Weight decay: 0.01

---

## 8. Overfitting Test
### 목적
소량 데이터로 모델이 정상적으로 학습되는지 확인

### 방법
- Train / Validation 분리
- Epoch 단위 성능 및 loss 비교

### 결과
- Train loss 지속 감소
- Validation loss 변화량 매우 작음
- **과적합 없이 학습 파이프라인 정상 작동 확인**

📌 Full Data 학습 가능하다고 판단

---

## 9. Full Data 학습
- Train + Validation 데이터를 통합하여 전체 데이터 학습
- 동일한 모델 구조 및 하이퍼파라미터 유지
- 최종 모델 저장 후 Inference 및 웹 서비스 적용

---

## 10. Inference 및 웹앱 데모
- Streamlit 기반 웹앱 구현
- 사용자 입력 댓글 → 실시간 가짜/진짜 판별
- 평균 Latency: 약 20~35ms
- Stability: 100%

---

## 11. 결론 및 성과
### 정량적 성과
- Train Loss: 0.0012
- Validation Loss: 0.00119
- Test 기준 F1-score: 1.00

### 정성적 성과
- 데이터 수집 → 전처리 → 학습 → 추론 → 웹앱 배포까지  
  **AI 모델 개발 전체 사이클 경험**

### 기술적 의의
- 가짜 리뷰·댓글 자동 탐지 가능성 확인
- 데이터 확장 시 다양한 도메인 적용 가능

---

## 12. 보완할 부분 및 향후 과제
- Validation loss 추가 감소 필요
- 가짜 텍스트 데이터 추가 수집
- Threshold 조정 및 하이퍼파라미터 튜닝을 통한 Precision 개선

---
