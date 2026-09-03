import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

control = [5,6,7,5,8,6,7,5,6,7]
treatment = [10,12,11,13,12,10,14,11,12,13]

t, p = ttest_ind(treatment, control)

print("Control Mean:", np.mean(control))
print("Treatment Mean:", np.mean(treatment))
print("P-value:", round(p,4))

if p < 0.05:
    print("Reject H0")
else:
    print("Fail to Reject H0")

plt.bar(["Control","Treatment"],
        [np.mean(control),np.mean(treatment)])
plt.show()
