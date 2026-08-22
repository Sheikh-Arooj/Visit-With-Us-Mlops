
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from the repository data folder
DATA_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATA_PATH)
print("Original dataset shape:", df.shape)

# Remove the unnecessary CustomerID column
# Remove unnecessary columns
df = df.drop(columns=["CustomerID", "Unnamed: 0"])

print("Dataset shape after removing unnecessary columns:", df.shape)

# Separate features and target variable
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the training and testing data locally
X_train.to_csv("tourism_project/model_building/X_train.csv", index=False)
X_test.to_csv("tourism_project/model_building/X_test.csv", index=False)
y_train.to_csv("tourism_project/model_building/y_train.csv", index=False)
y_test.to_csv("tourism_project/model_building/y_test.csv", index=False)

print("\nData preparation completed successfully.")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
