import pandas as pd

# Load the data 
df = pd.read_csv("data/hotel_bookings.csv")

# 1. Size and column types
print("Shape (rows, columns):", df.shape)
print(df.dtypes)

# 2. The target: how many bookings were cancelled?
print("\nOverall cancellation rate:", round(df["is_canceled"].mean(), 3))
print("\nCancellation rate by hotel:")
print(df.groupby("hotel")["is_canceled"].mean().round(3))

# 3. Bookings per year and hotel
print("\nBookings per year and hotel:")
print(pd.crosstab(df["arrival_date_year"], df["hotel"]))

# 4. Missing values
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

# 5. LEAKAGE CHECK: compare reservation_status with the target
print("\nreservation_status vs is_canceled:")
print(pd.crosstab(df["reservation_status"], df["is_canceled"]))

# 6. Look at a few real rows
print(df.head(5).T)
