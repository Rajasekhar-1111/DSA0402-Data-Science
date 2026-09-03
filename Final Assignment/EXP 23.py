import numpy as np
from scipy.stats import ttest_ind

# Example data - replace with actual A/B test data
design_A = np.array([
    0.12, 0.15, 0.11, 0.14, 0.13,
    0.16, 0.12, 0.14, 0.15, 0.13
])

design_B = np.array([
    0.18, 0.20, 0.17, 0.19, 0.21,
    0.18, 0.20, 0.19, 0.17, 0.22
])


mean_A = np.mean(design_A)
mean_B = np.mean(design_B)

t_stat, p_value = ttest_ind(
    design_A,
    design_B
)

print("Mean Conversion Rate - A:",
      round(mean_A, 4))

print("Mean Conversion Rate - B:",
      round(mean_B, 4))

print("T-statistic:",
      round(t_stat, 4))

print("P-value:",
      round(p_value, 4))


if p_value < 0.05:
    print("Reject H0")
    print("There is a significant difference.")
else:
    print("Fail to Reject H0")
    print("There is no significant difference.")
