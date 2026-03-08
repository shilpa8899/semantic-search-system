from sklearn.datasets import fetch_20newsgroups


def load_dataset():
    """
    Load the 20 Newsgroups dataset.

    We remove headers, footers, and quoted text because
    they contain metadata and email signatures that do not
    contribute meaningful semantic information for clustering.
    """
    print("Downloading / Loading dataset...")

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    documents = dataset.data
    labels = dataset.target
    categories = dataset.target_names

    return documents, labels, categories


if __name__ == "__main__":

    docs, labels, categories = load_dataset()

    print("Total documents:", len(docs))
    print("Number of categories:", len(categories))

    print("\nCategories:\n")
    for c in categories:
        print("-", c)

    print("\nExample document:\n")
    print(docs[0][:500])