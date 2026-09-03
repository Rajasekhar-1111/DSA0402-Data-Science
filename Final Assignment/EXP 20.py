import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

df = pd.read_csv("data.csv")

stop_words = {
    "the", "and", "is", "a", "an", "to",
    "of", "in", "for", "this", "that", "it"
}

text = " ".join(df["feedback"].astype(str)).lower()

text = re.sub(r"[^a-z\s]", "", text)

words = text.split()

words = [w for w in words if w not in stop_words]

frequency = Counter(words)

n = int(input("Enter N: "))

top_words = frequency.most_common(n)

print("\nTop", n, "Words:")

for word, count in top_words:
    print(word, ":", count)

words_list = [x[0] for x in top_words]
counts = [x[1] for x in top_words]

plt.bar(words_list, counts)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top Frequent Words")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
