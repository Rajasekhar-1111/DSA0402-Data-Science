import pandas as pd

# Customer purchase data
data = {
    "Customer_ID": [101, 102, 103, 104, 105, 106, 107, 108,
                    109, 110, 111, 112, 113, 114, 115],
    "Age": [22, 25, 30, 22, 35, 25, 28, 30,
            22, 40, 35, 25, 28, 30, 22]
}

# Create DataFrame
df = pd.DataFrame(data)

print("CUSTOMER PURCHASE DATA")
print(df)

# Frequency distribution of ages
frequency = df["Age"].value_counts().sort_index()

print("\nFREQUENCY DISTRIBUTION OF CUSTOMER AGES")
print(frequency)

# Display as a table
result = frequency.reset_index()
result.columns = ["Age", "Frequency"]

print("\nAGE FREQUENCY TABLE")
print(result)
