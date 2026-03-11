def reasoning_accuracy(df):

    correct = 0

    for _, row in df.iterrows():

        response = row["response"]

        if row["ground_truth"] in response:
            correct += 1

    return correct / len(df)
