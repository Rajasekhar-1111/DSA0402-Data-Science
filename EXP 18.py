import pandas as pd

# Social media post data
data = {
    "Post_ID": [1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13, 14, 15],
    "Likes": [100, 150, 200, 100, 250, 150, 300, 200,
              100, 350, 250, 150, 200, 100, 300]
}

# Create DataFrame
df = pd.DataFrame(data)

print("SOCIAL MEDIA POST DATA")
print(df)

# Frequency distribution
frequency = df["Likes"].value_counts().sort_index()

print("\nFREQUENCY DISTRIBUTION OF LIKES")
print(frequency)

# Create result table
result = frequency.reset_index()
result.columns = ["Likes", "Frequency"]

print("\nLIKES FREQUENCY TABLE")
print(result)
