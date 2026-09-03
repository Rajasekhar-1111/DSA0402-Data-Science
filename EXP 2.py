import numpy as np

# 3x3 matrix (Sales price of products)
sales_data = np.array([
    [120, 150, 180],
    [200, 220, 210],
    [90, 100, 110]
])

# Average price
average_price = np.mean(sales_data)

print("Sales Data:")
print(sales_data)

print("\nAverage Price of All Products:", average_price)
