import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(r"C:\Users\megha\PycharmProjects\databyte\train.csv")

profile = ProfileReport(
    df,
    title="Train Data Profiling Report",
    explorative=True
)


profile.to_file("train_data_profiling.html")
