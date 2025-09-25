import streamlit as st
import pandas as pd
import pickle
import numpy as np
# from surprise import SVDpp
import logging
from pathlib import Path

# =====================
# Logging
# =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================
# Config
# =====================
DATA_FILE = "final_amazon_dataset.csv"
SCORE_MODEL_FILE = "score_model.pkl"
# PRODUCT_MODEL_FILE removed / not used for category-aware CF predictions

# =====================
# Load Data & Model
# =====================
@st.cache_data
def load_data():
    if not Path(DATA_FILE).exists():
        st.error(f"Dataset file '{DATA_FILE}' not found!")
        st.stop()
    df = pd.read_csv(DATA_FILE)
    # Ensure string types for IDs
    df['UserId'] = df['UserId'].astype(str)
    df['ProductId'] = df['ProductId'].astype(str)
    required_columns = ['UserId', 'ProductId', 'Score', 'product_name', 'Brand', 'main_category']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    return df

@st.cache_resource
def load_model(path):
    if not Path(path).exists():
        st.error(f"Model file '{path}' not found!")
        st.stop()
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
score_model = load_model(SCORE_MODEL_FILE)

# =====================
# Helper Functions
# =====================
def get_user_id_from_input(user_input):
    if not user_input:
        return None
    user_input = str(user_input)
    if user_input in df['UserId'].values:
        return user_input
    # try profile columns
    profile_columns = ['profile_name', 'ProfileName', 'user_name', 'UserName', 'name', 'Name', 'ProfileName']
    for col in profile_columns:
        if col in df.columns:
            matched = df[df[col].astype(str).str.lower() == user_input.lower()]
            if not matched.empty:
                return str(matched['UserId'].iloc[0])
    return None

def get_product_id_from_input(product_input):
    if not product_input:
        return None
    product_input = str(product_input)
    if product_input in df['ProductId'].values:
        return product_input
    # try exact product name
    if 'product_name' in df.columns:
        matched = df[df['product_name'].astype(str).str.lower() == product_input.lower()]
        if not matched.empty:
            return str(matched['ProductId'].iloc[0])
        # partial match
        matched = df[df['product_name'].astype(str).str.lower().str.contains(product_input.lower(), na=False)]
        if not matched.empty:
            return str(matched['ProductId'].iloc[0])
    return None

def validate_user_product(user_input, product_input):
    user_id = get_user_id_from_input(user_input)
    product_id = get_product_id_from_input(product_input)
    user_exists = user_id in df['UserId'].astype(str).values if user_id else False
    product_exists = product_id in df['ProductId'].astype(str).values if product_id else False
    return user_exists, product_exists, user_id, product_id

def predict_cf_for_pair(user_id, product_id):
    """Return predicted rating (float) for given user_id and product_id using score_model."""
    try:
        return score_model.predict(str(user_id), str(product_id)).est
    except Exception as e:
        logger.error(f"CF predict error for ({user_id}, {product_id}): {e}")
        return np.nan

# =====================
# Category-aware CF Top-N
# =====================
def recommend_in_category_cf(user_id, category, top_n=5, min_ratings=1):
    """
    Recommend top-N products in `category` for user_id using collaborative model predictions.
    - Filters out products the user already rated.
    - Uses the score_model to predict rating for each candidate.
    - Returns a DataFrame with columns: ProductId, product_name, brand, avg_score, num_ratings, pred_score
    """
    try:
        user_id = str(user_id)
        # candidates in category
        candidates = df[df['main_category'] == category].copy()
        if candidates.empty:
            return pd.DataFrame()

        # aggregate rating stats for display (avg and count)
        agg = candidates.groupby('ProductId').agg(
            avg_score=('Score', 'mean'),
            num_ratings=('Score', 'count'),
            product_name=('product_name', 'first'),
            brand=('Brand', 'first')
        ).reset_index()

        # filter by min_ratings if requested
        if min_ratings > 1:
            agg = agg[agg['num_ratings'] >= min_ratings]
            if agg.empty:
                return pd.DataFrame()

        # exclude items the user already rated
        user_rated = set(df[df['UserId'] == user_id]['ProductId'].astype(str).unique())
        agg = agg[~agg['ProductId'].astype(str).isin(user_rated)]

        if agg.empty:
            return pd.DataFrame()

        # predict for candidates using score_model
        preds = []
        # To avoid very long loops on huge categories, we can sample or limit candidates
        # but here we predict for all remaining candidates (should be fine for moderate sizes)
        for pid in agg['ProductId'].astype(str).unique():
            pred_score = predict_cf_for_pair(user_id, pid)
            if np.isnan(pred_score):
                continue
            preds.append((pid, pred_score))

        if not preds:
            return pd.DataFrame()

        # build DataFrame with predicted scores
        preds_df = pd.DataFrame(preds, columns=['ProductId', 'pred_score'])
        merged = preds_df.merge(agg, on='ProductId', how='left')

        # rank by predicted score then by number of ratings (tie-breaker)
        merged = merged.sort_values(['pred_score', 'num_ratings'], ascending=[False, False])

        # limit to top_n
        merged = merged.head(top_n)

        # round numeric columns for nicer display
        merged['pred_score'] = merged['pred_score'].round(3)
        merged['avg_score'] = merged['avg_score'].round(3)

        return merged[['ProductId', 'product_name', 'brand', 'avg_score', 'num_ratings', 'pred_score']].reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error in recommend_in_category_cf: {e}")
        return pd.DataFrame()

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Amazon Recommendation System", page_icon="📦", layout="wide")
st.title("📦 Amazon Product Recommendation System")
st.markdown("---")

st.sidebar.page_link("pages/Chat_Bot.py", label="💬 Chatbot")

# Sidebar
st.sidebar.title("🎯 Recommendation Options")
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("Category-Based (CF)",  "Popularity", "Hybrid" ,"Dataset Overview")
)
# Sidebar dataset info
with st.sidebar.expander("📊 Dataset Info"):
    st.write(f"**Total Products:** {df['ProductId'].nunique():,}")
    st.write(f"**Total Users:** {df['UserId'].nunique():,}")
    st.write(f"**Total Ratings:** {len(df):,}")
    st.write(f"**Average Rating:** {df['Score'].mean():.2f}")

col1, col2 = st.columns([2, 1])

# ---------------------
# Inputs (shared)
# ---------------------
with col1:
    if option not in ["Dataset Overview", "Popularity"]:
        st.subheader("🔍 Input Parameters")
        # user input
        profile_cols = ['profile_name','ProfileName','user_name','UserName','name','Name']
        has_profile = any(c in df.columns for c in profile_cols)
        user_label = "Profile Name or User ID" if has_profile else "User ID"
        sample_users = df['UserId'].unique()[:50].tolist()
        if has_profile:
            profile_col = next((c for c in profile_cols if c in df.columns), None)
            sample_profile_names = df[profile_col].dropna().unique()[:20].tolist()
            sample_users = list(sample_profile_names) + sample_users

        user_input = st.selectbox(f"Select {user_label}", options=[''] + sample_users)
        if not user_input:
            user_input = st.text_input(f"Or enter {user_label} manually", value=str(df['UserId'].iloc[0]))

        # product/category input
        sample_product_names = df['product_name'].dropna().unique()[:50].tolist()
        product_input = st.selectbox("Select Product Name or ID (optional)", options=[''] + sample_product_names)
        if not product_input:
            product_input = st.text_input("Or enter Product Name or ID manually", value=str(df['ProductId'].iloc[0]))

        # category chooser (fallback)
        categories = df['main_category'].dropna().unique().tolist()
        chosen_category = st.selectbox("Or pick a category (optional)", options=[''] + categories)

# resolve ids
if option not in ["Dataset Overview","Popularity"]:
    user_id = get_user_id_from_input(user_input) if user_input else None
    product_id = get_product_id_from_input(product_input) if product_input else None

    with col2:
        if user_id:
            st.write(f"**User resolved:** {user_id}")
        if product_id:
            st.write(f"**Product resolved:** {product_id}")
        if user_id and product_id:
            user_exists, product_exists, _, _ = validate_user_product(user_input, product_input)
            st.info("**Validation Status:**")
            st.write(f"User exists: {'✅' if user_exists else '❌'}")
            st.write(f"Product exists: {'✅' if product_exists else '❌'}")

st.markdown("---")

# ---------------------
# Category-based using CF predictions (TOP-N by predicted rating)
# ---------------------
if option == "Category-Based (CF)":
    st.subheader("📦 Category-based recommendations (using collaborative predictions)")
    top_n = st.slider("Number of recommendations", 1, 20, 5)
    min_ratings = st.slider("Min ratings required for candidate product", 1, 100, 1)

    if st.button("🎯 Get Category Recommendations", type="primary"):
        # determine category: priority -> product_input's category -> chosen_category -> error
        category_to_use = None
        if product_id:
            row = df[df['ProductId'] == str(product_id)]
            if not row.empty:
                category_to_use = row['main_category'].iloc[0]
        if not category_to_use and chosen_category:
            category_to_use = chosen_category
        if not category_to_use:
            st.error("Please provide a product (to infer its category) or explicitly pick a category.")
        elif not user_id:
            st.error("Please provide a valid User ID or Profile Name.")
        else:
            with st.spinner("Predicting and ranking candidates..."):
                recs_df = recommend_in_category_cf(user_id, category_to_use, top_n=top_n, min_ratings=min_ratings)
            if recs_df.empty:
                st.warning("No recommendations found (maybe user has rated all items in this category or no candidates met min_ratings).")
            else:
                st.success(f"Top {len(recs_df)} in category '{category_to_use}' for user {user_id}:")
                st.table(recs_df)


# ---------------------
# Popularity
# ---------------------
elif option == "Popularity":
    st.subheader("🔥 Trending Products")
    top_n = st.slider("Number of products to show",5,50,10)
    if st.button("📈 Show Trending Products", type="primary"):
        pop_df = df.groupby('ProductId').agg(
            avg_score=('Score','mean'), num_ratings=('Score','count'),
            product_name=('product_name','first'), brand=('Brand','first'), main_category=('main_category','first')
        ).reset_index()
        pop_df = pop_df[pop_df['num_ratings'] >= max(1, int(df.groupby('ProductId').size().quantile(0.7)))]
        pop_df = pop_df.sort_values(['avg_score','num_ratings'], ascending=[False, False]).head(top_n)
        st.table(pop_df[['ProductId','product_name','brand','main_category','avg_score','num_ratings']])

# ---------------------
# Hybrid
# ---------------------
elif option == "Hybrid":
    st.subheader("⚖️ Hybrid Recommendation (CF + Category popularity)")
    alpha = st.slider("CF weight (alpha)", 0.0, 1.0, 0.7, 0.05)
    top_n = st.slider("Number of results", 1, 20, 5)
    if st.button("🎯 Get Hybrid Recommendations", type="primary"):
        # Hybrid: compute CF prediction and combine with normalized avg rating in category
        category_to_use = None
        if product_id:
            row = df[df['ProductId'] == str(product_id)]
            if not row.empty:
                category_to_use = row['main_category'].iloc[0]
        if not category_to_use and chosen_category:
            category_to_use = chosen_category
        if not category_to_use:
            st.error("Provide a product (to infer category) or pick a category.")
        elif not user_id:
            st.error("Please provide a valid User ID or Profile Name.")
        else:
            cand = df[df['main_category'] == category_to_use].groupby('ProductId').agg(
                avg_score=('Score','mean'), num_ratings=('Score','count'),
                product_name=('product_name','first'), brand=('Brand','first')
            ).reset_index()
            cand = cand[~cand['ProductId'].isin(df[df['UserId'] == user_id]['ProductId'])]
            if cand.empty:
                st.warning("No candidates found after excluding user's rated items.")
            else:
                # compute predictions
                preds = []
                for pid in cand['ProductId'].astype(str).unique():
                    p = predict_cf_for_pair(user_id, pid)
                    if np.isnan(p):
                        continue
                    preds.append((pid, p))
                if not preds:
                    st.warning("No CF predictions available for candidates.")
                else:
                    preds_df = pd.DataFrame(preds, columns=['ProductId','cf_pred'])
                    merged = preds_df.merge(cand, on='ProductId', how='left')
                    # normalize avg_score to 0-1
                    if merged['avg_score'].isnull().all():
                        merged['avg_norm'] = 0.0
                    else:
                        min_a = merged['avg_score'].min()
                        max_a = merged['avg_score'].max()
                        if max_a - min_a == 0:
                            merged['avg_norm'] = 1.0
                        else:
                            merged['avg_norm'] = (merged['avg_score'] - min_a) / (max_a - min_a)
                    merged['final_score'] = alpha * merged['cf_pred'] + (1-alpha) * (merged['avg_norm'] * 5.0)
                    merged = merged.sort_values('final_score', ascending=False).head(top_n)
                    merged['final_score'] = merged['final_score'].round(3)
                    st.table(merged[['ProductId','product_name','brand','cf_pred','avg_score','num_ratings','final_score']])

# ---------------------
# Dataset Overview
# ---------------------
elif option == "Dataset Overview":
    st.subheader("📊 Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Products", f"{df['ProductId'].nunique():,}")
    c2.metric("Total Users", f"{df['UserId'].nunique():,}")
    c3.metric("Total Ratings", f"{len(df):,}")
    c4.metric("Avg Rating", f"{df['Score'].mean():.2f}")
    st.subheader("📈 Rating Distribution")
    st.bar_chart(df['Score'].value_counts().sort_index())
    st.subheader("🏷️ Top Product Categories")
    st.bar_chart(df['main_category'].value_counts().head(10))
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(10))
