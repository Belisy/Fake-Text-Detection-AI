import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 학습된 모델 위치, 주소는 세션(런터임) 다시 시작될 때마다 매번 달라짐
model_path = "/content/results/checkpoint-639"

model = AutoModelForSequenceClassification.from_pretrained(model_path)


# -----------------------------
# 1) 모델 로드
# -----------------------------
# model_path = "./results/checkpoint-final"  # 학습된 모델 위치


model_name = "beomi/KcELECTRA-base-v2022"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# -----------------------------
# 2) Streamlit UI 구성
# -----------------------------
st.title("🧠 AI 기반 댓글 판별기")
st.write("입력한 댓글이 **진짜인지**, **AI가 생성한 가짜인지** 확률과 함께 알려드립니다.")

# session_state로 새로고침 전까지 기록 유지
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# 3) 댓글 입력창
# -----------------------------
user_input = st.text_input("댓글을 입력하세요:", "")

# -----------------------------
# 4) 댓글 판별 함수
# -----------------------------
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return probs  # [fake_prob, real_prob]

# -----------------------------
# 5) 예측 실행
# -----------------------------
if user_input:
    probs = predict(user_input)
    fake_prob = probs[0]
    real_prob = probs[1]

    st.subheader("📊 판별 결과")
    st.write(f"✔ **진짜 댓글일 확률:** {real_prob*100:.2f}%")
    st.write(f"✔ **AI 생성(가짜) 댓글일 확률:** {fake_prob*100:.2f}%")

    # 간단한 설명 제공 (조건3)
    st.subheader("📝 판별 이유(간단 설명)")
    if real_prob > fake_prob:
        st.write("이 댓글은 자연스러운 표현과 문장 구조를 가지고 있어 진짜일 가능성이 높습니다.")
    else:
        st.write("이 댓글은 반복적이거나 전형적인 문장 패턴을 사용해 AI 생성 가능성이 높습니다.")

    # 입력 기록 저장
    st.session_state.history.append({
        "text": user_input,
        "fake": fake_prob,
        "real": real_prob
    })

# -----------------------------
# 6) 기록 보여주기
# -----------------------------
st.subheader("🗂 입력했던 댓글 기록 (새로고침 전까지 유지)")
for item in st.session_state.history:
    st.write(f"- {item['text']} → 진짜:{item['real']:.2f}, 가짜:{item['fake']:.2f}")
