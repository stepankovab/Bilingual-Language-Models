from datasets import load_dataset


def load_limited_babylm(dataset_name = "BabyLM-community/BabyLM-2026-Strict-Small", max_words: int = 9_000_000, seed: int = 42):
    """
    Approximate word-based truncation (BabyLM = ~words, not tokens).
    """

    ds = load_dataset(dataset_name, split="train")
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]

    ds = ds.shuffle(seed=seed)

    selected = []
    total_words = 0

    for ex in ds:
        txt = ex[text_col]
        wc = len(txt.split())

        if total_words + wc > max_words:
            break

        selected.append(ex)
        total_words += wc

    return ds.select(range(len(selected)))