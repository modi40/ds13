import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sales.csv')

plt.figure(figsize=(8, 6))
plt.boxplot(df['Sales'], patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))
plt.title("Daily Sales for Two Months")
plt.ylabel("Sales Amount ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['Sales'] < lower) | (df['Sales'] > upper)]
print("Detected Outliers:")
print(outliers)
