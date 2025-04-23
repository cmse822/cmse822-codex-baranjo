import pandas as pd
import matplotlib.pyplot as plt
import os
# Check if the CSV files exist
print(os.getcwd())

# Load the two CSV files
# Make sure each CSV has columns named "x" and "y"
df1 = pd.read_csv("cmse822-codex-baranjo/projects/project4/jacsolv_gpu.csv")  # e.g., GPU
df2 = pd.read_csv("cmse822-codex-baranjo/projects/project4/jacsolv_cpu.csv")  # e.g., CPU

# Create the plot
plt.figure(figsize=(10, 6))
plt.loglog(df1['x'], df1['y'], marker='o', label='GPU', linestyle='-')
plt.loglog(df2['x'], df2['y'], marker='s', label='CPU (best thread count)', linestyle='--')

# Add labels, title, legend, and grid
plt.xlabel("Matrix Dimension")
plt.ylabel("FLOP/s")
plt.title("Jacobi Solver Performance: GPU vs CPU (Best)")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot to a file
plt.savefig("jacsolv_comparison.png", dpi=300)
