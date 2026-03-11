from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def hallucination_rate(df, threshold=0.5):

    hallucinations = 0

    for _, row in df.iterrows():

        gt = row["ground_truth"]
        response = row["response"]

        emb1 = model.encode(gt, convert_to_tensor=True)
        emb2 = model.encode(response, convert_to_tensor=True)

        similarity = util.cos_sim(emb1, emb2).item()

        if similarity < threshold:
            hallucinations += 1

    return hallucinations / len(df)
