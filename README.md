# 🛒 E-commerce Recommendation Engine

This project is a **Recommendation Engine** built with collaborative filtering (SVD), content-based filtering, and hybrid models.  
It helps users discover relevant products based on their browsing history, ratings, and product similarities.

## 🚀 Live Demo
Try the application here:  
👉 [E-commerce Recommendation Engine](https://ecommerce-recommendation-engine.streamlit.app/)

## 📦 Features
- Collaborative Filtering (SVD)  
- Content-Based Filtering  
- Hybrid Recommendations  
- Sidebar filters for product name, price, and ratings  

## 🔄 Project Workflow
1. **Data Preprocessing**  
   - Enhance Amazon review dataset with realistic product names and categories.  
   - Clean and prepare data for recommendation models.  

2. **Exploratory Analysis**  
   - Compute dataset statistics (users, products, sparsity).  
   - Analyze user activity and product popularity.  

3. **Model Development**  
   - **Collaborative Filtering (SVD):** Predict ratings based on user-item interactions.  
   - **Content-Based Filtering:** Recommend similar products using product features (name, category).  
   - **Hybrid Model:** Combine both approaches for improved recommendations.  

4. **Streamlit Application**  
   - User can select product or user ID.  
   - Apply filters (price, rating, category).  
   - Display top-N recommendations in a clean interface.  

## 📊 Tech Stack
- Python, Pandas, Numpy  
- Scikit-learn  
- Streamlit  
