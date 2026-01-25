import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Change the path to YOUR OWN local CSV file
df = pd.read_csv("/Users/alexandrathudor/Documents/Deep Learning Project/task_adaptation_dl/CASP.csv")

df.head()

# Basic inspection
df.shape
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.nunique()


#Descriptive statistics
mean_values = df.mean()
median_values = df.median()
mode_values = df.mode().iloc[0]
std_dev = df.std()
variance = df.var()
correlation_matrix = df.corr()
covariance_matrix = df.cov()
skewness = df.skew()
kurtosis = df.kurtosis()
summary_stats = pd.DataFrame({
    'Mean': df.mean(),
    'Median': df.median(),
    'Std Dev': df.std(),
    'Variance': df.var(),
    'Skewness': df.skew(),
    'Kurtosis': df.kurtosis()
})

print(summary_stats)




# Spearman correlation matrix
spearman_corr = df.corr(method='spearman')

plt.figure(figsize=(8,6))
sns.heatmap(spearman_corr, cmap='coolwarm')
plt.title("Spearman Correlation Matrix")
plt.show()



# PCA (to explore redundancy)
X = df.drop(columns=['RMSD'])
X_scaled = StandardScaler().fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.show()


# Box plot
X_log = df.drop(columns='RMSD').copy()
X_log[['F5','F7']] = np.log1p(X_log[['F5','F7']]) # Log-transform skewed features

plt.figure(figsize=(12,6))
sns.boxplot(data=X_log)
plt.xticks(rotation=45) # yassified for better readability
plt.ylabel("Log-transformed feature values")
plt.title("Boxplots of Log-Transformed Predictor Features")
plt.show()