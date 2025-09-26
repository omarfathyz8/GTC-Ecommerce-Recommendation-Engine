# 📦 Amazon Recommendation System

This project implements a **Hybrid Recommendation System** that combines **Collaborative Filtering** with **Popularity-Based** and **Category-Aware** filtering to provide personalized Amazon product recommendations based on user ratings and product metadata.

---

## 🚀 Project Workflow

### 1. Data Preprocessing
- ✅ Handled missing values in user and product data.
- ✅ Verified no duplicate user-product rating pairs.
- ✅ Fixed data types (UserId, ProductId as strings).
- ✅ Validated required columns: `UserId`, `ProductId`, `Score`, `product_name`, `Brand`, `main_category`.

### 2. Feature Engineering
- Ensured consistent string formatting for IDs.
- Created aggregated product statistics:
  - `avg_score` = Average rating per product
  - `num_ratings` = Total ratings per product
  - `rating_std` = Rating variance per product
- Built user behavior profiles:
  - User average rating patterns
  - User activity levels (number of ratings given)
- **Text Processing** (for content-based models):
  - Combined `Text` + `Summary` → `text_all`
  - Applied TF-IDF vectorization
  - Created product similarity matrices

### 3. Exploratory Data Analysis (EDA)
- 📊 **Rating Distribution**: Most ratings are positive (4-5 stars)
- 👥 **User Behavior**: Wide range of user activity levels
- 📦 **Product Categories**: Electronics and Books dominate the dataset
- ⭐ **Quality Products**: High-rated products tend to have more reviews
- 🏷️ **Brand Analysis**: Popular brands show consistent rating patterns

---

## 🧩 Models

### 1. Collaborative Filtering (CF)
- Algorithm: **SVD** (Singular Value Decomposition) from scikit-surprise
- Model File: `score_model.pkl` (trained CF model)
- Features used: `(UserId, ProductId, Score)`
- Core Function:
  ```python
  def predict_cf_for_pair(user_id, product_id):
      return score_model.predict(str(user_id), str(product_id)).est
  ```
- **Performance Metrics**:
  - RMSE: Optimized through hyperparameter tuning
  - MAE: Validated on test set
- **Real-time Predictions**: Individual user-product rating predictions
- **Category-Aware**: CF predictions within specific product categories

---

### 2. Content-Based Filtering

#### A. Text-Based Model 
- **TF-IDF Vectorization** of product descriptions and reviews
- **Cosine Similarity** for product-to-product recommendations
- Features: `Text`, `Summary`, combined into `text_all`
- **Nearest Neighbors** system for finding similar products

#### B. Category-Based Content Filtering
- Uses product metadata: `main_category`, `Brand`
- Filters candidates within same category
- Combines with CF scores for hybrid recommendations

---

### 3. Matrix Factorization Models

#### B. SVD++ (Enhanced SVD - Not Currently Active)
- **Advanced Model**: Incorporates implicit feedback
- Accounts for user rating patterns beyond explicit scores
- Higher computational cost but potentially better accuracy
- Code prepared but commented out in current deployment:
  ```python

---

### 4. Hybrid Models

#### A. Weighted Hybrid (CF + Popularity)
- **Primary Hybrid**: Currently deployed
- Formula: `α * CF_score + (1-α) * normalized_popularity`
- Adjustable weighting parameter (α) via slider
- Combines personalization with popularity trends

#### B. Category-CF Hybrid
- **Category + CF**: Currently deployed
- Filters by category, ranks by CF predictions
- Ensures recommendations within user's interest areas
- Fallback to popularity when CF fails

---

### 5. Popularity-Based Models
- **Trending Products**: Global popularity ranking
- Metrics: Average rating × Number of reviews
- **Threshold Filtering**: Minimum rating requirements (70th percentile)
- **Category-Specific**: Popular items within categories
---

## 🖥️ Deployment
- Built **interactive Streamlit interface** with multiple recommendation modes
- **Smart Input Resolution**: Handles user names, profile names, or IDs
- **Real-time Validation**: Instant feedback on user/product existence
- **Rich Analytics Dashboard**: Dataset statistics and visualizations
- Integrated **Chatbot** for enhanced user experience

---

## 🎯 Features

### 🔍 Recommendation Types
- **Category-Based (CF)**: Personalized recommendations within categories
- **Rating Prediction**: Predict individual user-product ratings
- **Popularity**: Discover trending and highly-rated products
- **Hybrid**: Combined approach with adjustable weighting
- **Dataset Overview**: Interactive data exploration

### 🎨 User Interface
- **Dropdown Menus**: Easy selection from sample users/products
- **Manual Entry**: Direct ID or name input with smart matching
- **Validation Panel**: Real-time input verification
- **Results Display**: Clean tables with detailed metrics
- **Statistical Context**: Product and user behavior insights

---

## ⚙️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, Scikit-surprise)
- **ML Models**: SVD, SVD++, Matrix Factorization
- **NLP**: TF-IDF vectorization, text preprocessing (for content-based features)
- **Similarity Metrics**: Cosine similarity, Nearest Neighbors
- **Visualization**: Streamlit charts and metrics
- **Deployment**: Streamlit web interface
- **Caching**: Efficient data and model loading

---

## 📂 Project Structure
```
├── final_amazon_dataset.csv    # Main dataset
├── score_model.pkl            # Trained CF model
├── app.py                     # Main Streamlit application
├── pages/
│   └── Chat_Bot.py           # Chatbot integration
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```

---


## 📊 Dataset Requirements

Your CSV should contain:
- `UserId`: User identifier
- `ProductId`: Product identifier  
- `Score`: Rating (1-5 scale)
- `product_name`: Product name
- `Brand`: Product brand
- `main_category`: Product category

Optional (for user-friendly names):
- `profile_name`, `ProfileName`, `user_name`, `UserName`, `name`, `Name`

---

## ✨ Key Features

- **Smart Input Resolution**: Automatic matching of names to IDs
- **Real-time Validation**: Instant feedback on input validity
- **Rich Context**: Product details, statistics, and user insights
- **Configurable Parameters**: Adjustable recommendation counts and thresholds
- **Interactive Interface**: User-friendly Streamlit design
- **Performance Optimized**: Efficient caching and data processing

---
