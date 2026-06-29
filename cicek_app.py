import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================
# Sayfa Ayarları
# =========================
st.set_page_config(
    page_title="Çiçek Türü Tanıma Sistemi",
    page_icon="🌸",
    layout="wide"
)

# =========================
# CSS Tasarım
# =========================
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2e7d32;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.info-card {
    background-color: #f1f8e9;
    padding: 18px;
    border-radius: 15px;
    border-left: 6px solid #66bb6a;
    margin-bottom: 15px;
}

.warning-card {
    background-color: #fff8e1;
    padding: 18px;
    border-radius: 15px;
    border-left: 6px solid #ffb300;
    margin-bottom: 15px;
}

.result-card {
    background-color: #e8f5e9;
    padding: 22px;
    border-radius: 18px;
    border: 2px solid #81c784;
    text-align: center;
    margin-top: 15px;
}

.result-title {
    font-size: 28px;
    font-weight: bold;
    color: #1b5e20;
}

.confidence-text {
    font-size: 22px;
    color: #2e7d32;
}

.small-text {
    color: #666;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Başlık
# =========================
st.markdown('<div class="main-title">🌸 Görüntüden Çiçek Türü Tanıma Sistemi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Yapay zeka modeli, yüklenen çiçek fotoğrafını analiz ederek türünü tahmin eder.</div>',
    unsafe_allow_html=True
)

# =========================
# Model ve Sınıf İsimlerini Yükleme
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("flower_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names.json", "r", encoding="utf-8") as f:
        return json.load(f)

try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error("Model veya sınıf dosyası yüklenemedi.")
    st.code(str(e))
    st.stop()

# =========================
# Türkçe isimler
# =========================
turkish_names = {
    "daisy": "Papatya",
    "dandelion": "Karahindiba",
    "roses": "Gül",
    "sunflowers": "Ayçiçeği",
    "tulips": "Lale"
}

flower_emojis = {
    "daisy": "🌼",
    "dandelion": "🌾",
    "roses": "🌹",
    "sunflowers": "🌻",
    "tulips": "🌷"
}

# =========================
# Yan Menü
# =========================
with st.sidebar:
    st.header("📌 Proje Bilgileri")

    st.markdown("""
    Bu sistem **derin öğrenme** ve **hazır eğitilmiş model** kullanarak çiçek türü tahmini yapar.
    """)

    st.divider()

    st.subheader("Tanıyabildiği Çiçekler")

    for class_name in class_names:
        tr_name = turkish_names.get(class_name, class_name)
        emoji = flower_emojis.get(class_name, "🌸")
        st.write(f"{emoji} {tr_name}")

    st.divider()

    st.subheader("Model Bilgisi")
    st.write("Model dosyası: `flower_model.keras`")
    st.write("Girdi boyutu: `224 x 224`")
    st.write("Sınıf sayısı:", len(class_names))

    st.divider()

    st.info("Not: Sistem sadece yukarıdaki 5 çiçek türünü tanıyacak şekilde eğitilmiştir.")

# =========================
# Ana Alan
# =========================
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("📤 Çiçek Görseli Yükle")
    st.write("JPG, JPEG veya PNG formatında bir çiçek fotoğrafı seç.")
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Görsel seç",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Yüklenen Görsel", width="stretch")

with right_col:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🤖 Tahmin Sonucu")
    st.write("Görsel yüklendikten sonra model tahmin sonucunu burada gösterecek.")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.warning("Lütfen önce bir çiçek görseli yükleyin.")

    else:
        if st.button("🔍 Tahmin Et", use_container_width=True):
            with st.spinner("Model görseli analiz ediyor..."):
                # Görseli modelin istediği boyuta getir
                img = image.resize((224, 224))
                img_array = np.array(img)

                # Batch boyutu ekle
                img_array = np.expand_dims(img_array, axis=0)

                # Tahmin
                predictions = model.predict(img_array)[0]

                # Büyükten küçüğe sırala
                sorted_indices = np.argsort(predictions)[::-1]

                best_index = sorted_indices[0]
                second_index = sorted_indices[1]

                best_class = class_names[best_index]
                second_class = class_names[second_index]

                best_name_tr = turkish_names.get(best_class, best_class)
                second_name_tr = turkish_names.get(second_class, second_class)

                best_emoji = flower_emojis.get(best_class, "🌸")
                second_emoji = flower_emojis.get(second_class, "🌸")

                best_confidence = predictions[best_index] * 100
                second_confidence = predictions[second_index] * 100

                difference = best_confidence - second_confidence

                # Güven kontrolü
                if best_confidence < 70 or difference < 15:
                    st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                    st.warning("Model bu görselde tam emin değil.")
                    st.write(f"En yakın tahmin: **{best_emoji} {best_name_tr}**")
                    st.write(f"Güven oranı: **%{best_confidence:.2f}**")
                    st.write(f"İkinci tahmin: **{second_emoji} {second_name_tr} - %{second_confidence:.2f}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="result-title">{best_emoji} Tahmin Edilen Çiçek: {best_name_tr}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div class="confidence-text">Güven Oranı: %{best_confidence:.2f}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                st.divider()

                # İlk 3 tahmin
                st.subheader("🏆 En Yüksek 3 Tahmin")

                for rank, index in enumerate(sorted_indices[:3], start=1):
                    class_name = class_names[index]
                    class_tr = turkish_names.get(class_name, class_name)
                    emoji = flower_emojis.get(class_name, "🌸")
                    probability = predictions[index] * 100

                    st.write(f"{rank}. {emoji} **{class_tr}** - %{probability:.2f}")
                    st.progress(float(predictions[index]))

                st.divider()

                # Tüm olasılıklar
                with st.expander("📊 Tüm tahmin olasılıklarını göster"):
                    for i, class_name in enumerate(class_names):
                        class_tr = turkish_names.get(class_name, class_name)
                        emoji = flower_emojis.get(class_name, "🌸")
                        probability = predictions[i] * 100

                        st.write(f"{emoji} {class_tr}: %{probability:.2f}")
                        st.progress(float(predictions[i]))

