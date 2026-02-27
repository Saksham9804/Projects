# 📊 Data Science & ML Projects

A collection of three end-to-end machine learning and data analytics projects built in Python using Google Colab.

---

## 📁 Projects Overview

| Project | Type | Key Libraries |
|---|---|---|
| Market Analysis | Data Analytics + ML | pandas, sklearn, matplotlib, seaborn |
| Movie Recommendation System | Content-Based Filtering | sklearn, NLTK, cosine similarity |
| Vehicle Classification | Deep Learning (CNN) | PyTorch, ResNet18, torchvision |

---

## 📌 Project 1: Market Analysis & Sales Insights

**File:** `Data_Analytics.ipynb`

### Description
An end-to-end data analytics project on a supermarket sales dataset, covering data cleaning, exploratory data analysis (EDA), machine learning models, and automated PDF report generation.

### Dataset
Downloaded from Kaggle: `sakshamtiwari98/market-analysis` — a supermarket transaction dataset with columns for branch, city, customer type, product line, payment method, ratings, and more.

### Workflow

**Data Cleaning & Preprocessing**
- Converted word-form quantities (e.g., "Six" → 6) to numeric values
- Parsed and standardized `Date` and `Time` columns
- Dropped rows with missing critical fields
- Engineered new features: `DayOfWeek`, `Hour`, `TimeOfDay`

**Exploratory Data Analysis (EDA)**
- Revenue by branch and product line
- Average transaction value per city
- Spend breakdown by customer type and gender
- Payment method preferences
- Peak shopping hours histogram
- Correlation matrix and scatter plots
- Gross income and profit-per-unit by product line

**Machine Learning**
- **Regression** — Linear Regression to predict customer ratings using unit price, quantity, and gross income (evaluated with RMSE and R²)
- **Classification** — Random Forest Classifier to predict customer type (Member vs Normal), using encoded features including time of day, day of week, payment method, and more

**Output**
- Summarized EDA answers exported to a `Part 3 Answers.pdf` using `reportlab`

### Key Libraries
```
pandas, numpy, matplotlib, seaborn, sklearn, reportlab, kagglehub
```

---

## 📌 Project 2: Movie Recommendation System

**File:** `Movie_Prediction.ipynb`

### Description
A content-based movie recommendation system built on the TMDB 5000 Movies dataset. The system recommends the top 10 most similar movies to any given title using text feature engineering and cosine similarity.

### Dataset
Downloaded from Kaggle: `tmdb/tmdb-movie-metadata` — contains two CSV files (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`) with movie metadata including genres, keywords, cast, crew, and overviews.

### Workflow

**Data Loading & Merging**
- Merged the movies and credits datasets on the `title` column
- Retained: `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`

**Feature Extraction**
- Parsed stringified JSON columns (`genres`, `keywords`, `cast`, `crew`) using `ast.literal_eval`
- Extracted top 3 cast members per movie
- Extracted the director from the crew list
- Combined all features into a single `tags` text column per movie

**Text Preprocessing**
- Lowercased all text
- Removed English stopwords using NLTK
- Applied Porter Stemming to normalize word forms

**Vectorization & Similarity**
- Used `CountVectorizer` with `max_features=5000` to convert tags to bag-of-words vectors
- Computed pairwise cosine similarity across all movies

**Recommendation**
- Given a movie title, the system ranks all others by similarity score and returns the top 10 matches

### Example
```
Input:  Spider-Man 3
Output: Spider-Man 2, Spider-Man, The Amazing Spider-Man 2, The Amazing Spider-Man, Arachnophobia, ...
```

### Key Libraries
```
pandas, numpy, sklearn, nltk, ast, kagglehub
```

---

## 📌 Project 3: Vehicle Image Classification

**File:** `Vehicle_Detection.ipynb`

### Description
A deep learning image classification project that trains a fine-tuned ResNet18 model to identify vehicle types from images. The model is trained using transfer learning on a multi-class vehicle dataset.

### Dataset
Downloaded from Kaggle: `mohamedmaher5/vehicle-classification` — a folder-structured dataset with images organized by vehicle class.

### Workflow

**Data Preparation**
- Split each class into train (70%), validation (15%), and test (15%) sets using `train_test_split`
- Reorganized images into a directory structure compatible with PyTorch's `ImageFolder`

**Data Transforms**
- Resized all images to 224×224
- Applied random horizontal flips and rotations for training augmentation
- Normalized using ImageNet mean and std values

**Model Architecture**
- Pre-trained **ResNet18** (ImageNet weights) from `torchvision.models`
- Replaced the final fully connected layer to match the number of vehicle classes
- Trained using **CrossEntropyLoss** and the **Adam** optimizer with gradient clipping

**Training**
- Trained for 5 epochs with separate train and validation phases
- Best model saved based on highest validation accuracy

**Evaluation**
- Reports overall test accuracy
- Reports per-class accuracy for each vehicle category

**Prediction**
- `predict_image()` function accepts any image path and returns the predicted vehicle class
- Visualization of one sample per class with true and predicted labels side by side

### Key Libraries
```
torch, torchvision, sklearn, numpy, matplotlib, PIL, kagglehub
```

---

## 🚀 Getting Started

All notebooks are designed to run in **Google Colab** with Kaggle dataset integration.

### Prerequisites
```bash
pip install kagglehub reportlab nltk
```

Set up your Kaggle API credentials before running any notebook:
1. Go to [kaggle.com](https://www.kaggle.com) → Account → Create API Token
2. Upload the `kaggle.json` file to your Colab session or configure it via environment variables

### Running a Notebook
1. Open the notebook in Google Colab using the badge at the top of each file
2. Run all cells from top to bottom
3. For `Vehicle_Detection.ipynb`, GPU runtime is recommended (Runtime → Change runtime type → GPU)

---

## 📂 Repository Structure

```
Projects/
├── Data_Analytics.ipynb        # Market sales EDA + ML + PDF report
├── Movie_Prediction.ipynb      # Content-based movie recommender
├── Vehicle_Detection.ipynb     # CNN vehicle image classifier
└── README.md
```

---

## 👤 Author

**Saksham Tiwari**  
[GitHub](https://github.com/Saksham9804)
