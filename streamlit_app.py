# streamlit_app.py
import streamlit as st
import time

import os
from PIL import Image

# فرض کنیم یه تصویر confusion matrix تو پوشه پروژه ذخیره شده
image_path = "confusionMats\modifiedCodeTestData"  # مسیر تصویر




# ---------- تنظیمات اولیه ----------
st.set_page_config(page_title="GCNN Cancer Classifier", layout="centered")
st.title("🧬 طبقه‌بندی سرطان با GCNN")
st.markdown("**پروژه کارشناسی | بازتولید مقاله با اجرای مدل‌های مختلف**")
st.markdown("---")

# ---------- انتخاب نسخه کد ----------
st.subheader("🧩 ۱. انتخاب نسخه کد")

code_version = st.radio(
    "کدام نسخه از کد مدل را می‌خواهید اجرا کنید؟",
    ["نسخه اصلی مقاله", "نسخه اصلاح‌شده (این پروژه)"]
)

# ---------- انتخاب گراف ----------
st.subheader("🧠 ۲. انتخاب نوع گراف زیستی")

graph_option = st.selectbox(
    "نوع گراف مورد استفاده:",
    ["PPI", "PPIS", "COEX", "COEXS"]
)

# ---------- اجرای مدل ----------
st.subheader("🚀 ۳. اجرای مدل")

if st.button("🏁 اجرای مدل"):
    st.info("⏳ در حال اجرای مدل GCNN...")
    time.sleep(2)  # شبیه‌سازی زمان اجرا

    # -------------------------
    # اجرای مدل واقعی بر اساس گزینه‌ها
    # -------------------------
    def run_gcnn_model(graph_type, version):
        # در اینجا کد واقعی خودت رو جایگزین کن
        # مثلاً:
        if version == "نسخه اصلی مقاله":
            # return original_model_run(graph_type)
            if graph_type == "PPI":
                return {
                    "accuracy": 93.2,
                    "std": 0.025
                }
            if graph_type == "PPIS":
                return {
                    "accuracy": 93.2,
                    "std": 0.025
                }
            if graph_type == "COEX":
                return {
                    "accuracy": 93.2,
                    "std": 0.025
                }
            if graph_type == "COEXS":
                return {
                    "accuracy": 93.2,
                    "std": 0.025
                }
        else:
            # return modified_model_run(graph_type)
            if graph_type == "PPI":
                return {
                    "accuracy": 93.2,
                    "std": 0.025
                }
            if graph_type == "PPIS":
                return {
                    "accuracy": 93.81,
                    "std": 0.66
                }
            if graph_type == "COEX":
                return {
                    "accuracy": 93.2,
                    "std": 0.025
                }
            if graph_type == "COEXS":
                return {
                    "accuracy": 93.2,
                    "std": 0.025
                }

    results = run_gcnn_model(graph_option, code_version)

    # ---------- نمایش نتایج ----------
    st.success("✅ مدل با موفقیت اجرا شد!")
    col1, col2 = st.columns(2)
    col1.metric("🎯 دقت مدل", f"{results['accuracy']}%")
    col2.metric("📊 انحراف معیار", f"± {results['std']}")
    if os.path.exists(image_path):
        st.subheader("🖼️ ماتریس سردرگمی مدل")
        image = Image.open(image_path+"\CM_Mod"+graph_option+".png")
        st.image(image, caption="Confusion Matrix for PPI Graph", use_column_width=True)
    else:
        st.warning("تصویر ماتریس سردرگمی یافت نشد. لطفاً بررسی کنید که فایل در مسیر مشخص‌شده وجود دارد.")
    #st.markdown(f"📌 **نوع سرطان پیش‌بینی‌شده:** *{results['predicted_class']}*")

# ---------- فوتر ----------
st.markdown("---")
st.markdown("🧑‍💻 توسعه: مبینا هادوی فر | استاد راهنما: دکتر حقیرچهرقانی")
