import numpy as np

# 4x4 matrix
student_scores = np.array([
    [85, 90, 78, 88],
    [76, 85, 80, 79],
    [90, 88, 92, 91],
    [70, 75, 68, 72]
])

subjects = ["Math", "Science", "English", "History"]

# Average of each subject (column-wise)
avg_scores = np.mean(student_scores, axis=0)

print("Average Score of Each Subject:")
for i in range(len(subjects)):
    print(subjects[i], ":", avg_scores[i])

# Subject with highest average
highest = np.argmax(avg_scores)

print("\nSubject with Highest Average:", subjects[highest])
print("Highest Average Score:", avg_scores[highest])
