import pandas as pd

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/processed_hourly.csv"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

# ---------------- COLUMN NAMES ----------------
print("\nColumn Names:")
print(df.columns.tolist())

# ---------------- DATETIME HANDLING ----------------
# Handle datetime column safely (case-insensitive)
datetime_col = None
for col in df.columns:
    if col.lower() == "datetime":
        datetime_col = col
        break

if datetime_col:
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    print(f"\nDatetime column '{datetime_col}' converted successfully.")
    print("Date Range:")
    print(df[datetime_col].min(), "to", df[datetime_col].max())
else:
    print("\n⚠️ No datetime column found!")

# ---------------- DATA TYPES ----------------
print("\nData Types:")
print(df.dtypes)

# ---------------- SAMPLE DATA ----------------
print("\nFirst 5 Rows:")
print(df.head())

# ---------------- MISSING VALUES ----------------
print("\nMissing Values (per column):")
print(df.isnull().sum())
