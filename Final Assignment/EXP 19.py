import pandas as pd
from collections import Counter
import re

# Customer reviews dataset
data = {
    "Review": [
        "Good product and good quality",
        "Excellent product and good service",
        "Good quality product",
        "Excellent service and good product",
        "Product quality is good",
        "Good product with excellent quality"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

print("CUSTOMER REVIEWS")
print(df)

# Combine all reviews
text = " ".join(df["Review"])

# Convert to lowercase
text = text.lower()

# Remove punctuation
text = re.sub(r"[^a-z\s]", "", text)

# Split into words
words = text.split()

# Calculate word frequency
word_frequency = Counter(words)

print("\nWORD FREQUENCY DISTRIBUTION")

for word, count in word_frequency.most_common():
    print(word, ":", count)
