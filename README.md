# 📦 Amazon Recommendation System

This project implements a **Hybrid Recommendation System** that combines **Collaborative Filtering (SVD)** with **Content-Based Filtering** (using both textual data and product categories). The goal is to improve product recommendations based on user reviews, ratings, and item metadata.  

---

## 🚀 Project Workflow  

### 1. Data Preprocessing  
- ✅ Handled missing values.  
- ✅ Verified no duplicate entries.  
- ✅ Fixed data types.  

### 2. Feature Engineering  
- Extracted time-based features: `year`, `month`, `day`, `dayofweek`.  
- Created new features:  
  - `helpfulness_ratio = HelpfulnessNumerator / HelpfulnessDenominator`  
  - `time_weight = 1 / (1 + (year - df["year"]))`  
  - `final_score`, `final_score2`, `final_score3` (weighted variations of score).  

### 3. Exploratory Data Analysis (EDA)  
- 📈 Reviews increased significantly over the years.  
- ⭐ 60% of ratings are **5**, showing a strong positive skew.  
- 👥 4 users are clear outliers with much higher review counts.  
- 📦 One product has a significantly higher number of reviews.  
- 📅 Ratings dip mid-week and rise on weekends.  
- 🏆 **1999–2000** best years for ratings, **2001** worst.  
- 📊 Post-2012: stable but fluctuating ratings trend.  

---

## 🧩 Models  

### 1. Collaborative Filtering (CF)  🤖 [Download Model](https://drive.google.com/file/d/1Mbv1LRy1gxP5jzH8zDCbUML7JBn3JVzn/view?usp=sharing "Text-Based Content Filtering Model")
- Algorithm: **SVD** (after hyperparameter tuning).  
- Best hyperparameters:  
  ```python
  {'n_factors': 50, 'n_epochs': 80, 'lr_all': 0.02, 'reg_all': 0.08}
  ```
- Features used: `(UserID, ProductID, final_score2)`  
- Results:  
  - RMSE = **0.9334**  
  - MAE  = **0.8112**  

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

#### A. SVD++ (Enhanced SVD )
- **Primary Model**: Currently deployed in the system
- Decomposes user-item interaction matrix
- Handles sparse data effectively
- Captures latent factors in user preferences
---

### 4. Hybrid Models

#### A. Weighted Hybrid (CF + Popularity)
- **Primary Hybrid**: Currently deployed
- Formula: `α * CF_score + (1-α) * normalized_popularity`
- Adjustable weighting parameter (α) via slider
- Combines personalization with popularity trends

#### B. Content-CF Hybrid (Not Currently Deployed)
- **Text + Summary + CF**: Weighted combination
- Dynamic α based on user activity levels
- More active users → higher CF weight
- Less active users → higher content-based weight

#### C. Category-CF Hybrid
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
- **Time-Weighted**: Could incorporate recency (future enhancement)

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


## ✨ Key Features

- **Smart Input Resolution**: Automatic matching of names to IDs
- **Real-time Validation**: Instant feedback on input validity
- **Rich Context**: Product details, statistics, and user insights
- **Configurable Parameters**: Adjustable recommendation counts and thresholds
- **Interactive Interface**: User-friendly Streamlit design
- **Performance Optimized**: Efficient caching and data processing

---

## 📥 Model Downloads
- 🤖 **[SVD Collaborative Filtering Model](https://drive.google.com/file/d/1Mc_O7_L5xIiAU02LTyEVI67RafMuRHwz/view?usp=sharing)** - Main recommendation engine
- 🤖 **[Text-Based Content Model](https://drive.google.com/file/d/1Mbv1LRy1gxP5jzH8zDCbUML7JBn3JVzn/view?usp=sharing)** - Content-based filtering model

