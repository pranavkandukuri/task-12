# task-12

AI & ML Internship – Task 12
KMeans – Customer Segmentation

1. Introduction
Customer segmentation is an unsupervised learning technique used to group customers based on similar characteristics.
In this task, KMeans clustering is applied to segment mall customers using their Annual Income and Spending Score.
This helps businesses understand customer behavior and target them effectively.

2. Tools Used
•	Python
•	Scikit-learn
•	Pandas
•	Matplotlib

3. Dataset
•	Primary Dataset: Mall Customer Segmentation Dataset (Kaggle)
•	Important columns:
o	CustomerID
o	Annual Income (k$)
o	Spending Score (1-100)

4. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

5. Load Dataset
df = pd.read_csv("Mall_Customers.csv")
df.head()

6. Inspect Dataset
df.info()

7. Drop CustomerID Column
df = df.drop('CustomerID', axis=1)
Reason:
CustomerID does not contribute to clustering.

8. Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

9. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Reason:
KMeans uses distance calculations, so scaling is required for balanced clustering.

10. Elbow Method to Find Optimal K
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

11. Elbow Curve Plot
plt.figure()
plt.plot(range(1,11), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()
Observation:
The point where the curve bends indicates the optimal number of clusters.

12. Train KMeans Model
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df.head()

13. Cluster Visualization
plt.figure()
plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using KMeans")
plt.show()

14. Save Segmented Dataset
df.to_csv("segmented_customers.csv", index=False)

15. Cluster Interpretation
•	Cluster 0: High income – high spending (Premium customers)
•	Cluster 1: Low income – low spending
•	Cluster 2: High income – low spending
•	Cluster 3: Low income – high spending
•	Cluster 4: Average income – average spending

16. Final Outcome
•	Learned unsupervised learning using KMeans
•	Used Elbow Method to find optimal clusters
•	Visualized customer segments
•	Created a segmented dataset for business use

