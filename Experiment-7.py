import pandas as pd

order_data = pd.DataFrame({
    "Customer_ID": [101, 102, 101, 103, 102],
    "Order_Date": ["2024-01-10", "2024-01-12", "2024-01-15", "2024-01-18", "2024-01-20"],
    "Product": ["Laptop", "Mobile", "Laptop", "Tablet", "Mobile"],
    "Order_Quantity": [2, 1, 3, 2, 4]
})

print("Total Orders by Customer")
print(order_data.groupby("Customer_ID").size())

print("\nAverage Order Quantity")
print(order_data.groupby("Product")["Order_Quantity"].mean())

print("\nEarliest Order Date")
print(order_data["Order_Date"].min())

print("\nLatest Order Date")
print(order_data["Order_Date"].max())