 
# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\Hp\Downloads\Titanic-Dataset.csv")

# Step 3: Display First 5 Rows
print("First 5 Rows")
print(df.head())

# Step 4: Display Dataset Information
print("\nDataset Information")
print(df.info())

# Step 5: Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Step 6: Fill Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# ==========================================
# Visualization 1: Survival Count
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)

plt.title("Passenger Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

plt.show()

# ==========================================
# Visualization 2: Survival by Gender
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Survival by Gender")

plt.show()

# ==========================================
# Visualization 3: Passenger Class
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)

plt.title("Passenger Class Distribution")

plt.show()

# ==========================================
# Visualization 4: Age Distribution
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(df["Age"], bins=20, kde=True)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Count")

plt.show()

# ==========================================
# Visualization 5: Fare Distribution
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(df["Fare"], bins=20, kde=True)

plt.title("Fare Distribution")

plt.xlabel("Fare")

plt.ylabel("Count")

plt.show()

# Visualization 6: Correlation Heatmap

numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(8,6))

sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()

# Visualization 7: Pie Chart

survival = df["Survived"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    survival,
    labels=["Not Survived","Survived"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Survival Percentage")

plt.show()

# Final Insights

print("\n========== DATA VISUALIZATION COMPLETED ==========")
print("\nKey Insights:")
print("1. Female passengers survived more than male passengers.")
print("2. Most passengers travelled in Third Class.")
print("3. Most passengers were between 20 and 40 years old.")
print("4. Most passengers paid lower fares.")
print("5. Heatmap shows relationships between numerical columns.")
print("6. Pie chart shows the survival percentage.")