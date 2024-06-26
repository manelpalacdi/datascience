# a program that takes a dataset containing prices 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv(r"C:\Users\Manel\Desktop\CODE\PYTHON\datascience\product_analysis\data.csv")
    df = data_cleaning(df)
    data_plot(df, "P00001")
    

def data_cleaning(df: pd.DataFrame):
    # remove duplicates
    df = df.drop_duplicates()
    # replace "unknown" to NaN
    df = df.replace("unknown", np.nan)
    # remove all rows with NaN
    df = df.dropna(axis=0, how="any")
    # remove all rows with negative price or reviews (wrong input)
    for i in df.index:
        if (df.loc[i, "price"] < 0 or abs(df.loc[i, "reviews"]) > 5):
            df.drop(index=i, inplace=True)
    
    df = df.sort_values(by="date")

    return(df)        

def data_plot(df: pd.DataFrame, prod_id: str):
    df_prod = df[df["product_id"] == prod_id]

    plt.figure(figsize=(10, 6))

    # we get normalized values for the merged subplot
    norm_df_prod = df_prod.copy()
    norm_df_prod["price"] = (norm_df_prod["price"] - norm_df_prod["price"].min()) /  (norm_df_prod["price"].max() - norm_df_prod["price"].min())
    norm_df_prod["reviews"] = (norm_df_prod["reviews"] - norm_df_prod["reviews"].min()) /  (norm_df_prod["reviews"].max() - norm_df_prod["reviews"].min())
    norm_df_prod["sales"] = norm_df_prod["sales"].astype(float)
    norm_df_prod["sales"] = (norm_df_prod["sales"] - norm_df_prod["sales"].min()) /  (norm_df_prod["sales"].max() - norm_df_prod["sales"].min())

    plt.subplot(2, 1, 1)
    plt.plot(norm_df_prod["date"], norm_df_prod["price"], color="red", label="price", marker=".")
    plt.plot(norm_df_prod["date"], norm_df_prod["reviews"], color="green", label="reviews", marker=".")
    plt.plot(norm_df_prod["date"], norm_df_prod["sales"], color="orange", label="sales", marker=".")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.bar(df_prod["date"], df_prod["price"], alpha=0.5)
    plt.plot(df_prod["date"], df_prod["price"], label="price", color="red", marker=".")
    plt.xticks(df_prod["date"].iloc[::3])
    plt.legend(loc="upper right")

    plt.subplot(2, 3, 5)
    plt.bar(df_prod["date"], df_prod["reviews"].astype(float), alpha=0.5)
    plt.plot(df_prod["date"], df_prod["reviews"].astype(float), label="reviews", color="green", marker=".")
    plt.xticks(df_prod["date"].iloc[::3])
    plt.legend(loc="upper right")
    
    plt.subplot(2, 3, 6)
    plt.bar(df_prod["date"], df_prod["sales"].astype(float), alpha=0.5)
    plt.plot(df_prod["date"], df_prod["sales"].astype(float), label="sales", color="orange", marker=".")
    plt.xticks(df_prod["date"].iloc[::3])
    plt.legend(loc="upper right")

    # rotate date labels so they fit better
    plt.gcf().autofmt_xdate()
    plt.show()

if __name__ == "__main__":
    main()