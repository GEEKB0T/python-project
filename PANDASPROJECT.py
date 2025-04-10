import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 1. Load and Clean the Dataset
# ----------------------------------------
df = pd.read_csv("C:/Users/rajde/Desktop/dataset.csv")

# Clean column names (lowercase, strip spaces)
df.columns = df.columns.str.strip().str.lower()

# Display basic information
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# ----------------------------------------
# 1A. Exploratory Data Analysis (EDA)
# ----------------------------------------

# Shape of the dataset
print(f"\nDataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Summary statistics
print("Summary statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check unique values in categorical columns
print("\nUnique values in 'branch' and 'year':")
print("Branches:", df['branch'].unique())
print("Years:", df['year'].unique())

# Countplot of students per year
plt.figure(figsize=(8, 4))
sns.countplot(x='year', data=df, palette='Set2')
plt.title("Number of Students per Year")
plt.xlabel("Academic Year")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Countplot of students per branch
plt.figure(figsize=(10, 4))
sns.countplot(y='branch', data=df, palette='Set3', order=df['branch'].value_counts().index)
plt.title("Number of Students per Branch")
plt.xlabel("Count")
plt.ylabel("Branch")
plt.tight_layout()
plt.show()

# ----------------------------------------
# 2. Data Cleansing
# ----------------------------------------
# Remove rows with missing CGPA or invalid entries
df = df.dropna(subset=['cgpa', 'branch', 'year'])
df = df[df['cgpa'].between(0, 10)]  # Valid CGPA range

# ----------------------------------------
# 3. Heatmap: Average CGPA by Year and Branch
# ----------------------------------------
grouped = df.groupby(['year', 'branch'])['cgpa'].mean().unstack()

plt.figure(figsize=(12, 6))
sns.heatmap(grouped, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0.5, linecolor='gray')
plt.title("Heatmap: Average CGPA Across Years and Branches")
plt.ylabel("Academic Year")
plt.xlabel("Branch")
plt.tight_layout()
plt.show()

# ----------------------------------------
# 4. Top Performing Branches
# ----------------------------------------
top_branches = df.groupby('branch')['cgpa'].mean().sort_values(ascending=False)
print("\nTop Performing Branches by Average CGPA:")
print(top_branches)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_branches.index, y=top_branches.values, palette='viridis')
plt.xticks(rotation=45)
plt.ylabel("Average CGPA")
plt.title("Branch-wise Average CGPA")
plt.tight_layout()
plt.show()

# ----------------------------------------
# 5. Visualize CGPA Trends
# ----------------------------------------

# Histogram of CGPA
plt.figure(figsize=(10, 5))
sns.histplot(df['cgpa'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of CGPA")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# Boxplot by Branch
plt.figure(figsize=(12, 6))
sns.boxplot(x='branch', y='cgpa', data=df)
plt.xticks(rotation=45)
plt.title("CGPA Distribution by Branch")
plt.tight_layout()
plt.show()

# Boxplot by Year
plt.figure(figsize=(10, 6))
sns.boxplot(x='year', y='cgpa', data=df)
plt.title("CGPA Distribution by Year")
plt.tight_layout()
plt.show()

# ----------------------------------------
# 6. Detect Outliers
# ----------------------------------------
Q1 = df['cgpa'].quantile(0.25)
Q3 = df['cgpa'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['cgpa'] < lower_bound) | (df['cgpa'] > upper_bound)]
print(f"\nOutliers Detected (CGPA < {lower_bound:.2f} or > {upper_bound:.2f}):")
print(outliers[['name', 'branch', 'year', 'cgpa']])

# ----------------------------------------
# 7. Summary Report
# ----------------------------------------

# Top 10 Overall Performers
top10_overall = df.sort_values(by='cgpa', ascending=False).head(10)
print("\nTop 10 Performers Overall:")
print(top10_overall[['name', 'branch', 'year', 'cgpa']])

# Top 10 per Branch
print("\nTop 10 Performers per Branch:")
top_per_branch = df.groupby('branch').apply(lambda x: x.sort_values(by='cgpa', ascending=False).head(10)).reset_index(drop=True)
print(top_per_branch[['name', 'branch', 'year', 'cgpa']])

# ----------------------------------------
# 8. Donut Chart – Student Distribution by Branch
# ----------------------------------------
branch_counts = df['branch'].value_counts()

plt.figure(figsize=(8, 8))
colors = sns.color_palette('pastel')[0:len(branch_counts)]
plt.pie(branch_counts, labels=branch_counts.index, colors=colors, startangle=90,
        wedgeprops={'width': 0.4}, autopct='%1.1f%%')
plt.title("Distribution of Students by Branch")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

# ----------------------------------------
# 9. Correlation Heatmap (Numeric Features)
# ----------------------------------------
plt.figure(figsize=(6, 4))
numeric_corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(numeric_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap: Correlation Between Numeric Features")
plt.tight_layout()
plt.show()
