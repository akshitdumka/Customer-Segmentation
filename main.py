# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# Step 2: Load the Dataset
df = pd.read_csv('ifood_df.csv')  # Ensure the file is in your working directory
print("Dataset loaded successfully!")
print(df.head())
# Step 3: Explore and Clean the Data
print(df.info())
print(df.isnull().sum())

# Drop rows with missing values (optional: fill with median if needed)
df = df.dropna()

# Step 4: Feature Engineering (Create RFM features)
df['Monetary'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

df['Frequency'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                      'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)

# Recency is already a column in the dataset
rfm = df[['Recency', 'Frequency', 'Monetary']].copy()
# Step 5: Data Normalization
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
# Step 6: Determine Optimal Clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
# Step 7: Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # You can change clusters based on elbow curve
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Optional: Add Customer ID back for tracking
rfm['Customer_ID'] = df['ID']
# Step 8: Visualize Clusters
sns.set(style="whitegrid")

# Recency vs Monetary
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='Set2')
plt.title("Customer Segments (Recency vs Monetary)")
plt.show()

# Boxplot: Frequency by Segment
plt.figure(figsize=(8, 5))
sns.boxplot(data=rfm, x='Segment', y='Frequency')
plt.title("Frequency Distribution by Segment")
plt.show()
# Step 9: Segment Summary and Insights
summary = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'Customer_ID': 'count'
}).rename(columns={'Customer_ID': 'Customer_Count'}).round(2)

print("Segment Summary:")
print(summary)
# Step 10: Save Results (Optional)
rfm.to_csv("ifood_customer_segments.csv", index=False)
print("Segmented customer data saved to 'ifood_customer_segments.csv'")
