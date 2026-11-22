import pandas as pd

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# print(df.head())

# print(df.shape)

# df.info()

# print(df.isnull().sum())

# print(df["isFraud"].value_counts())

# print(df["isFraud"].value_counts(normalize=True) * 100)


# print(df["type"].value_counts())

# fraud = df[df["isFraud"] == 1]

# print(fraud["type"].value_counts())


# fraud_rate = df.groupby("type")["isFraud"].mean() * 100

# print(fraud_rate)

print(df["amount"].describe())
print()

normal = df[df["isFraud"] == 0]
fraud = df[df["isFraud"] == 1]

print("Normal Mean:", normal["amount"].mean())
print("Fraud Mean:", fraud["amount"].mean())

print("Normal Median:", normal["amount"].median())
print("Fraud Median:", fraud["amount"].median())