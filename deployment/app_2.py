import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

artifacts = joblib.load(r"C:\Users\nice\Desktop\projects\internship system recomdation\models\svd_full.pkl")
model = artifacts["model"]
trainset = artifacts["trainset"]
reader = artifacts["reader"]
df_stats = artifacts["df_stats"] 

st.title("🎯 Recommender System (SVD with Streamlit)")
user_id = st.text_input("🧑 Enter User ID:")
product_id = st.text_input("📦 Enter Product ID:")

if st.button("Predict Rating and the simialry products"):
    if user_id and product_id:
        try:
            
            pred = model.predict(user_id, product_id)
            raw_pred = pred.est  
            min_val = df_stats["min_val"]
            max_val = df_stats["max_val"]
            scaled_pred = 1 + 4 * ((raw_pred - min_val) / (max_val - min_val))
            final_rating = int(round(scaled_pred))
            final_rating = max(1, min(5, final_rating))
            st.success(f"⭐ Predicted Rating: {final_rating}")

            item_factors = model.qi
            try:
                item_inner_id = trainset.to_inner_iid(product_id)
                item_vector = item_factors[item_inner_id].reshape(1, -1)
                similarities =cosine_similarity(item_vector, item_factors)[0]
                similar_indices = similarities.argsort()[::-1][1:11] 
                similar_product_ids = [trainset.to_raw_iid(inner_id) for inner_id in similar_indices]
                similar_scores = similarities[similar_indices]

                st.subheader("🔍 Top 10 Similar Products")
                for pid, score in zip(similar_product_ids, similar_scores):
                    st.write(f"📦 Product: {pid} | 🔗 Similarity: {score:.3f}")

            except ValueError:
                st.warning("⚠️ Product ID غير موجود في الداتا")

        except Exception as e:
            st.error(f"🚨 Error during prediction: {str(e)}")
    else:
        st.warning("⚠️ Please enter both User ID and Product ID.")
