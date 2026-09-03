import numpy as np

# Columns: Bedrooms, Square Feet, Sale Price
house_data = np.array([
    [3, 1500, 250000],
    [5, 2200, 450000],
    [4, 1800, 320000],
    [6, 2800, 600000],
    [5, 2400, 500000]
])

# Houses with more than 4 bedrooms
houses = house_data[house_data[:, 0] > 4]

# Average sale price
average_price = np.mean(houses[:, 2])

print("Houses with More Than 4 Bedrooms:")
print(houses)

print("\nAverage Sale Price:", average_price)
