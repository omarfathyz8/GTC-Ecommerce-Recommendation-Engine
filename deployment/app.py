
import streamlit as st
import joblib
import numpy as np
import pandas as pd

artifacts = joblib.load(r"C:\Users\nice\Desktop\projects\internship system recomdation\models\svd_full.pkl")
model = artifacts["model"]
df_stats = artifacts["df_stats"] 
indices = np.load(r"C:\Users\nice\Desktop\projects\internship system recomdation\metrics\indices.npy")
product_ids = np.load(r"C:\Users\nice\Desktop\projects\internship system recomdation\metrics\proudict_id.npy", allow_pickle=True).tolist()
df = pd.read_csv(r"C:\Users\nice\Desktop\projects\internship system recomdation\data\final_data.csv")
user_ratings_count = df.groupby("UserId")["Score"].count().to_dict()


def get_alpha(user_id):
    n_ratings = user_ratings_count.get(user_id, 0)
    if n_ratings < 10:
        return 0.3
    elif n_ratings < 50:
        return 0.5
    else:
        return 0.7

def get_collab_score(user_id, product_id):
    try:
        pred = model.predict(user_id, product_id)
        raw_pred = pred.est  
        min_val = df_stats["min_val"]
        max_val = df_stats["max_val"]
        scaled_pred = 1 + 4 * ((raw_pred - min_val) / (max_val - min_val))
        final_rating = int(round(scaled_pred))
        final_rating = max(1, min(5, final_rating))
        return float(final_rating)
    except:
        return 4.1

def weighted_hybrid_fast(user_id, product_id, top_n=10):
    alpha = get_alpha(user_id)
    try:
        product_idx = product_ids.index(product_id)
        similar_indices = indices[product_idx][1:top_n*10] 
    except:
        similar_indices = []

    final_scores = []
    for idx in similar_indices:
        pid = product_ids[idx]
        score_collab = get_collab_score(user_id, pid)
        rank = list(similar_indices).index(idx)
        score_content = 1 - (rank / (len(similar_indices)))
        final_score = alpha * score_collab + (1 - alpha) * score_content
        final_scores.append((pid, final_score))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_products = [pid for pid, _ in final_scores[:top_n]]
    return recommended_products, alpha



st.set_page_config(
    page_title="Hybrid Recommender System",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #4B7BE5;
        font-size: 38px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: gray;
        font-size: 18px;
    }
    .card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .card h4 {
        margin: 0;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Hybrid Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a User ID and a Product ID to get personalized recommendations</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Input Parameters")
    user_id_input = st.text_input("User ID")
    product_id_input = st.text_input("Product ID")
    top_n = st.slider("Number of Recommendations", 1, 20, 10)
    st.markdown("### ⚙️ Instructions")
    st.markdown("1. Enter a valid User ID.\n2. Enter a Product ID.\n3. Adjust number of recommendations.\n4. Click the button below.")

if st.button("Get Recommendations"):
    if user_id_input and product_id_input:
        with st.spinner("Calculating recommendations..."):
            recs, alpha_used = weighted_hybrid_fast(user_id_input, product_id_input, top_n=top_n)
            final_rate = get_collab_score(user_id_input, product_id_input)

        st.markdown(f"### ✅ Alpha Used: {alpha_used}")
        st.markdown(f"### ⭐ Product Rating: {final_rate}")

        st.markdown(f"### Top {top_n} Recommendations:")
        for i, pid in enumerate(recs, 1):
            st.markdown(f"""
                <div class="card">
                    <h4>{i}. Product ID: {pid}</h4>
                    <p>Estimated Rating: {get_collab_score(user_id_input, pid):.1f}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter both User ID and Product ID.")
