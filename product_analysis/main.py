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

    plt.subplot(1, 3, 1)
    plt.plot(df_prod["date"], df_prod["price"], label="price")
    plt.xticks(df_prod["date"].iloc[::3])
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(df_prod["date"], df_prod["reviews"], label="reviews")
    plt.xticks(df_prod["date"].iloc[::3])
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(df_prod["date"], df_prod["sales"], label="sales")
    plt.xticks(df_prod["date"].iloc[::3])
    plt.legend()

    # rotate date labels so they fit better
    plt.gcf().autofmt_xdate()
    plt.show()

if __name__ == "__main__":
    main()