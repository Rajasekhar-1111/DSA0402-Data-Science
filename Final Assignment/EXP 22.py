import numpy as np

# Example data - replace with the values given by your faculty
drug = np.array([
    12, 15, 10, 14, 13, 16, 11, 15, 14, 12
])

placebo = np.array([
    5, 7, 6, 4, 8, 5, 6, 7, 5, 6
])


def confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(len(data))

    lower = mean - 1.96 * se
    upper = mean + 1.96 * se

    return mean, lower, upper


drug_mean, drug_lower, drug_upper = confidence_interval(drug)

placebo_mean, placebo_lower, placebo_upper = confidence_interval(placebo)


print("Drug Group")
print("Mean:", round(drug_mean, 2))
print("95% CI:", round(drug_lower, 2),
      "to", round(drug_upper, 2))

print("\nPlacebo Group")
print("Mean:", round(placebo_mean, 2))
print("95% CI:", round(placebo_lower, 2),
      "to", round(placebo_upper, 2))
