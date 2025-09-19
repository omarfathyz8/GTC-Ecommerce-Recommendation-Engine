import streamlit as st
import pandas as pd

st.set_page_config(page_title="E-commerce Recommendation Engine", layout="wide")
st.title("🛒 E-commerce Recommendation Engine")

# ======================
# Load dataset
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_cleaned.csv")
    return df

df = load_data()

# ======================
# Dummy recommendation functions
# (replace with your real models later)
# ======================
def recommend_by_user(user_id, n=5):
    """Dummy collaborative filtering: returns top n random products for a user."""
    if user_id not in df['user_id'].values:
        return df.sample(n)
    return df[df['user_id'] == user_id].sample(n) if len(df[df['user_id'] == user_id]) >= n else df.sample(n)

def recommend_by_product(product_name, n=5):
    """Dummy content-based: returns random products in the same category."""
    if product_name not in df['product_name'].values:
        return df.sample(n)
    category = df[df['product_name'] == product_name]['category'].iloc[0]
    same_cat = df[df['category'] == category]
    return same_cat.sample(n) if len(same_cat) >= n else df.sample(n)

# ======================
# Streamlit App
# ======================

st.write("Welcome! Get personalized or product-based recommendations with prices, discounts, and ratings.")

# Sidebar controls
st.sidebar.header("🔎 Choose Recommendation Type")
rec_type = st.sidebar.radio("Recommendation Based On:", ["User ID", "Product Search"])

# Sidebar filters
st.sidebar.header("⚙️ Filters")
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 4.0, 0.5)
max_price = st.sidebar.number_input("Max Discounted Price", value=5000)

# Input area
if rec_type == "User ID":
    user_id_input = st.text_input("Enter User ID:", "")
    if st.button("Get Recommendations"):
        if user_id_input:
            results = recommend_by_user(user_id_input, n=6)
        else:
            st.warning("Please enter a User ID.")
            results = None
else:
    product_input = st.text_input("Enter Product Name:", "")
    if st.button("Get Recommendations"):
        if product_input:
            results = recommend_by_product(product_input, n=6)
        else:
            st.warning("Please enter a Product Name.")
            results = None

# ======================
# Display Recommendations
# ======================
if 'results' in locals() and results is not None:

    # Apply filters
    try:
        results = results[
            (pd.to_numeric(results['rating'], errors='coerce') >= min_rating) &
            (pd.to_numeric(results['discounted_price'], errors='coerce') <= max_price)
        ]
    except Exception as e:
        st.error(f"❌ Error while filtering: {e}")
        st.stop()
        
        import requests

    if len(results) == 0:
        st.error("No products match your filters.")
    else:
        st.subheader("✨ Recommended Products")
        for i in range(0, len(results), 3):   # step 3
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(results):
                    row = results.iloc[i + j]
                    with col:
                        with st.container():
                            st.image(row['img_link'], use_container_width=True)
                            st.markdown(f"**[{row['product_name'][:100] + "..."}]({row['product_link']})**")
                            st.write(f"💰 **Discounted:** ₹{row['discounted_price']}")
                            st.write(f"💸 Actual: ₹{row['actual_price']}")
                            st.write(f"🏷️ {row['discount_percentage']}% off")
                            st.write(f"⭐ {row['rating']} ({row['rating_count']} reviews)")
                            st.caption(row['about_product'][:100] + "..." if isinstance(row['about_product'], str) else "")
