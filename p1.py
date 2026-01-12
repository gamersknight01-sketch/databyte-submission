import pandas as pd
from ydata_profiling import ProfileReport

# Load the training data
df = pd.read_csv(r"C:\Users\megha\PycharmProjects\databyte\train.csv")

# Generate profiling report
profile = ProfileReport(
    df,
    title="Train Data Profiling Report",
    explorative=True
)

# Save report to HTML
profile.to_file("train_data_profiling.html")
