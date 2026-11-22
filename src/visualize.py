import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# counts = df["isFraud"].value_counts().sort_index()

# plt.bar(["Normal", "Fraud"], counts.values)

# plt.title("Normal vs Fraud Transactions")
# plt.xlabel("Transaction Type")
# plt.ylabel("Number of Transactions")
# plt.yscale("log")
# plt.savefig("fraud_distribution.png", dpi=300, bbox_inches="tight")
# plt.show()




# fraud_rate = df.groupby("type")["isFraud"].mean() * 100

# plt.figure(figsize=(9, 5))

# plt.bar(fraud_rate.index, fraud_rate.values)

# plt.title("Fraud Rate by Transaction Type")
# plt.xlabel("Transaction Type")
# plt.ylabel("Fraud Rate (%)")

# plt.show()

# plt.savefig("fraud_rate_by_type.png", dpi=300, bbox_inches="tight")


# plt.figure(figsize=(9, 5))

# plt.hist(df["amount"], bins=50)

# plt.title("Transaction Amount Distribution")
# plt.xlabel("Transaction Amount")
# plt.ylabel("Frequency")

# plt.show()



import numpy as np

normal = df[df["isFraud"] == 0]
fraud = df[df["isFraud"] == 1]

plt.figure(figsize=(10, 5))

plt.hist(
    np.log1p(normal["amount"]),
    bins=50,
    alpha=0.6,
    density=True,
    label="Normal"
)

plt.hist(
    np.log1p(fraud["amount"]),
    bins=50,
    alpha=0.6,
    density=True,
    label="Fraud"
)

plt.title("Fraud vs Normal Transaction Amount")
plt.xlabel("Log(1 + Amount)")
plt.ylabel("Density")
plt.legend()

plt.show()