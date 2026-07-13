import numpy as np

house_data = np.genfromtxt("house_data.csv", delimiter=",", skip_header=1)

houses = house_data[house_data[:, 0] > 4]

average = np.mean(houses[:, 2])

print("Average Sale Price:", average)
