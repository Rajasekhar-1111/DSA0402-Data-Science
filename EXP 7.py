import pandas as pd

data = {
    "Customer_ID": [101, 102, 101, 103, 102],
    "Order_Date": ["2024-01-10", "2024-01-12", "2024-01-15", "2024-01-18", "2024-01-20"],
    "Product": ["Laptop", "Mouse", "Laptop", "Keyboard", "Mouse"],
    "Quantity": [1, 2, 3, 1, 4]
}

order_data = pd.DataFrame(data)

print("Orders by Customer:")
print(order_data.groupby("Customer_ID").size())

print("\nAverage Quantity by Product:")
print(order_data.groupby("Product")["Quantity"].mean())

print("\nEarliest Order Date:", order_data["Order_Date"].min())
print("Latest Order Date:", order_data["Order_Date"].max())
