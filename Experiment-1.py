import numpy as np

student_scores = np.array([
    [80, 75, 90, 85],
    [70, 85, 88, 80],
    [90, 95, 85, 92],
    [85, 80, 78, 88]
])

subjects = ["Math", "Science", "English", "History"]

average = np.mean(student_scores, axis=0)

print("Average Scores:")
for i in range(4):
    print(subjects[i], ":", average[i])

highest = np.argmax(average)

print("Highest Average Subject:", subjects[highest])