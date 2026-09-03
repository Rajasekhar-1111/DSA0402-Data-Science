import numpy as np

sales_data = np.array([50000, 60000, 70000, 80000])

total_sales = np.sum(sales_data)
percent_increase = ((sales_data[3] - sales_data[0]) / sales_data[0]) * 100

print("Total Sales:", total_sales)
print("Percentage Increase:", percent_increase, "%")
