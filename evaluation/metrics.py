import pandas as pd

def compute_accuracy(df):

    correct = 0

    for _, row in df.iterrows():

        prediction = row["response"].lower()
        answer = row["ground_truth"].lower()

        if answer in prediction:
            correct += 1

    return correct / len(df)
