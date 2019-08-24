import pandas as pd


def f(df):
    # Copy dataframe to local scope.
    # In this case, the dataframe df in the global scope.
    # will not be modified
    df = df.copy()
    df.drop(columns=["a"], inplace=True)
    print(df)


def f2(df):
    # In this case the dataframe in the global scope
    # will be changed.
    df.drop(columns=["a"], inplace=True)
    print(df)

if __name__ == "__main__":
    d = pd.DataFrame({"a": [1111, 22222], "b": [33333, 44444]})
    print("First:")
    print(d)
    f(d)
    print(d)
    print("Second:")
    d2 = pd.DataFrame({"a": [1111, 22222], "b": [33333, 44444]})
    print(d2)
    f2(d2)
    print(d2)
