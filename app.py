import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# =========================================================
# CUSTOM DESIGN
# =========================================================
st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #cccccc;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD CNN MODEL
# =========================================================
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(
        "plant_disease_model.keras"
    )


model = load_trained_model()


# =========================================================
# LOAD CLASS NAMES
# =========================================================
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# =========================================================
# DISEASE INFORMATION
# =========================================================
disease_info = {

    "Bacterial spot": {
        "symptoms":
            "Small dark or water-soaked spots may appear "
            "on the leaves.",

        "advice":
            "Remove badly affected leaves, avoid overhead "
            "watering, and follow locally recommended "
            "disease-management practices."
    },

    "Early blight": {
        "symptoms":
            "Brown spots with ring-like patterns commonly "
            "appear on older leaves.",

        "advice":
            "Remove infected plant material, improve air "
            "circulation, and follow appropriate local "
            "crop-protection guidance."
    },

    "Late blight": {
        "symptoms":
            "Dark, irregular and water-soaked lesions may "
            "develop and spread quickly.",

        "advice":
            "Remove severely affected material and seek "
            "appropriate local agricultural guidance because "
            "late blight can spread rapidly."
    },

    "Leaf Mold": {
        "symptoms":
            "Yellow patches may appear on the upper leaf "
            "surface with mold-like growth underneath.",

        "advice":
            "Improve ventilation, reduce excessive humidity "
            "and remove heavily affected leaves."
    },

    "Septoria leaf spot": {
        "symptoms":
            "Numerous small circular spots with darker borders "
            "can appear on leaves.",

        "advice":
            "Remove affected leaves, keep foliage dry where "
            "possible and maintain good field sanitation."
    },

    "Spider mites Two-spotted spider mite": {
        "symptoms":
            "Leaves may develop tiny pale spots, yellowing "
            "or fine webbing.",

        "advice":
            "Inspect the underside of leaves and use locally "
            "appropriate mite-management practices."
    },

    "Target Spot": {
        "symptoms":
            "Brown lesions may develop with concentric "
            "ring-like patterns.",

        "advice":
            "Remove infected debris, improve airflow and "
            "follow recommended disease-management practices."
    },

    "Tomato Yellow Leaf Curl Virus": {
        "symptoms":
            "Leaves may curl upward, become yellow and show "
            "reduced growth.",

        "advice":
            "Remove severely infected plants and manage "
            "whitefly populations using appropriate local "
            "agricultural guidance."
    },

    "Tomato mosaic virus": {
        "symptoms":
            "Leaves may show mosaic-like light and dark green "
            "patterns or distortion.",

        "advice":
            "Remove infected plants, disinfect tools and avoid "
            "handling healthy plants immediately after infected ones."
    },

    "healthy": {
        "symptoms":
            "No disease pattern was detected by the model.",

        "advice":
            "Continue regular monitoring, proper watering, "
            "nutrition and general plant care."
    }
}


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">🌿 Plant Disease Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-Based Deep Learning System for Tomato Leaf Disease Detection'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header("🌱 About Project")

    st.write(
        "This application uses a Convolutional Neural "
        "Network (CNN) to classify tomato leaf images."
    )

    st.write("**Model:** CNN")
    st.write("**Classes:** 10")
    st.write("**Input Size:** 128 × 128")
    st.write("**Framework:** TensorFlow / Keras")

    st.divider()

    st.subheader("How to Use")

    st.write("1. Upload a tomato leaf image.")
    st.write("2. Click **Detect Disease**.")
    st.write("3. View disease and confidence.")
    st.write("4. Check the Top-3 predictions.")


# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "📤 Upload a Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is None:

    st.info(
        "👆 Upload a JPG, JPEG or PNG image to start detection."
    )


# =========================================================
# PREDICTION
# =========================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # LEFT SIDE - IMAGE
    # -----------------------------------------------------
    with col1:

        st.subheader("🍃 Uploaded Leaf")

        st.image(
            image,
            use_container_width=True
        )


    # -----------------------------------------------------
    # RIGHT SIDE - PREDICTION
    # -----------------------------------------------------
    with col2:

        st.subheader("🧠 AI Analysis")

        if st.button(
            "🔍 Detect Disease",
            type="primary",
            use_container_width=True
        ):

            # Resize
            img = image.resize((128, 128))

            # Convert to numpy array
            img_array = np.array(img)

            # Add batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # CNN prediction
            predictions = model.predict(
                img_array,
                verbose=0
            )[0]

            predicted_index = np.argmax(predictions)

            predicted_class = class_names[
                predicted_index
            ]

            confidence = (
                predictions[predicted_index] * 100
            )

            # Clean class name
            disease_name = (
                predicted_class
                .replace("Tomato___", "")
                .replace("_", " ")
            )


            # =================================================
            # MAIN RESULT
            # =================================================
            if "healthy" in disease_name.lower():

                st.success(
                    "✅ Healthy Tomato Leaf"
                )

            else:

                st.error(
                    f"⚠️ Disease Detected: {disease_name}"
                )


            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )


            # =================================================
            # DISEASE DETAILS
            # =================================================
            info = disease_info.get(
                disease_name,
                {
                    "symptoms":
                        "Specific symptom information is "
                        "not available.",

                    "advice":
                        "Consult an agricultural expert for "
                        "appropriate management guidance."
                }
            )

            st.subheader("🔎 Common Symptoms")

            st.write(
                info["symptoms"]
            )

            st.subheader("💡 General Management Advice")

            st.write(
                info["advice"]
            )


            # =================================================
            # TOP 3 PREDICTIONS
            # =================================================
            st.divider()

            st.subheader("📊 Top 3 Predictions")

            top_indices = np.argsort(
                predictions
            )[-3:][::-1]

            for index in top_indices:

                name = (
                    class_names[index]
                    .replace("Tomato___", "")
                    .replace("_", " ")
                )

                probability = (
                    predictions[index] * 100
                )

                st.write(
                    f"**{name} — {probability:.2f}%**"
                )

                st.progress(
                    min(int(probability), 100)
                )


            # =================================================
            # CONFIDENCE WARNING
            # =================================================
            if confidence < 60:

                st.warning(
                    "⚠️ The model has low confidence in this "
                    "prediction. Try a clearer image of the leaf."
                )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "🌱 Plant Disease Detection from Leaf Images | "
    "Deep Learning Project using CNN & Image Augmentation"
)