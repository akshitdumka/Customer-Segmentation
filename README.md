# 🛍️ Customer Segmentation Analysis (iFood Dataset)

## 📌 Project Overview
This project performs **Customer Segmentation Analysis** using the **iFood dataset (`ifood_df.csv`)**.  
The goal is to group customers based on their **purchase behavior** and **campaign responses**, so businesses can design **personalized marketing strategies**.

We apply the **RFM Model (Recency, Frequency, Monetary)** and **K-Means Clustering** to identify customer groups.

---

## ⚡ Key Steps

### 1. Data Collection
- Dataset: `ifood_df.csv`
- Contains customer demographics, purchase history, campaign responses.

### 2. Data Exploration & Cleaning
- Checked dataset structure (`.info()`, `.describe()`).
- Removed missing values.

### 3. Feature Engineering
- **Monetary**: Total customer spending across product categories.  
- **Frequency**: Number of campaigns accepted by a customer.  
- **Recency**: Days since last purchase (already available in dataset).  

### 4. Customer Segmentation (Clustering)
- Standardized RFM features using **StandardScaler**.
- Applied **K-Means clustering**.
- Determined optimal clusters using the **Elbow Method**.

### 5. Visualization
- Scatter plots (`Recency vs Monetary`).
- Boxplots (`Frequency by Segment`).
- Cluster distribution analysis.

### 6. Insights & Recommendations
Each customer cluster has different behavior:
- **Loyal Customers (High spending, low recency)** → Reward with loyalty programs.
- **Inactive Customers (High recency, low frequency)** → Reactivation campaigns.
- **New Customers (Low recency, low frequency)** → Encourage repeat purchases.
- **Potential Loyalists (Mid-high frequency & spending)** → Upselling strategies.

---

## 🛠️ Tech Stack
- **Python 3.x**
- **Libraries**:  
  - `pandas` – data manipulation  
  - `numpy` – numerical operations  
  - `matplotlib` & `seaborn` – visualization  
  - `scikit-learn` – clustering & preprocessing  

---

## ▶️ How to Run

1. Clone this repository or download the project.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

