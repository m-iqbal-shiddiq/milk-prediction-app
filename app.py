from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

from utils.model import choquet_integral, load_model, load_vgg16
from utils.image import extract_features_from_uploaded_file

# Define Model
vgg16_model = load_vgg16()
rf_model, xgb_model, lgbm_model = load_model()
choquet_weights = np.array([0.21672951, 0.0667899, 0.20563449])

st.title("Prediksi Produksi Susu Sapi")
st.write("Unggah gambar dan masukkan dimensi untuk memprediksi jumlah susu (liter).")

# File Uploaders
col1, col2 = st.columns(2)
with col1:
    side_image = st.file_uploader("📸 Gambar Samping", type=["jpg", "jpeg", "png"], key="side_img")
with col2:
    back_image = st.file_uploader("📸 Gambar Belakang", type=["jpg", "jpeg", "png"], key="back_img")
    
# Image Viewers
preview_col1, preview_col2 = st.columns(2)
with preview_col1:
    if side_image:
        st.markdown("**Preview Gambar Samping**")
        image = Image.open(side_image)
        st.image(image, width=200)  # Thumbnail size
with preview_col2:
    if back_image:
        st.markdown("**Preview Gambar Belakang**")
        image = Image.open(back_image)
        st.image(image, width=200)


# Numeric inputs
panjang = st.number_input("Panjang (cm)", min_value=0.0, format="%.2f", key="panjang")
lebar = st.number_input("Lebar (cm)", min_value=0.0, format="%.2f", key="lebar")
tinggi = st.number_input("Tinggi (cm)", min_value=0.0, format="%.2f", key="tinggi")


if st.button("Prediksi Susu (Liter)"):
    if (side_image is not None) and (back_image is not None):
        
        # # Extract features
        side_features = extract_features_from_uploaded_file(side_image, vgg16_model)
        back_features = extract_features_from_uploaded_file(back_image, vgg16_model) 
        
        # Format as DataFrames
        X_side_df = pd.DataFrame([side_features])
        X_back_df = pd.DataFrame([back_features])
        X_side_df.columns = [f"{i+1}s" for i in range(X_side_df.shape[1])]
        X_back_df.columns = [f"{i+1}b" for i in range(X_back_df.shape[1])]
        
        dim_df = pd.DataFrame([{
            "T Ambing dr Belakang": tinggi,
            "L Ambing dr Belakang": lebar,
            "Panjang Ambing dr samping": panjang
        }])
        
        X_combined_df = pd.concat([dim_df, X_side_df, X_back_df], axis=1)
        X = X_combined_df.values
        
        # Predict
        rf_pred = rf_model.predict(X)
        xgb_pred = xgb_model.predict(X)
        lgbm_pred = lgbm_model.predict(X)
        
        model_prediction = np.array([rf_pred[0], xgb_pred[0], lgbm_pred[0]])
        choquet_result = choquet_integral(choquet_weights, model_prediction)
        
        st.subheader("📊 Hasil Prediksi:")
        st.write(f"🔸 Random Forest: **{rf_pred[0]:.2f} liter**")
        st.write(f"🔸 XGBoost: **{xgb_pred[0]:.2f} liter**")
        st.write(f"🔸 LightGBM: **{lgbm_pred[0]:.2f} liter**")
        st.success(f"⭐ Prediksi Akhir (Choquet): **{choquet_result:.2f} liter**")
    else:
        st.warning("Mohon unggah kedua gambar: Gambar Samping dan Gambar Belakang.")
