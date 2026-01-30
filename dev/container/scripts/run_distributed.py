"""Generic distributed job demonstrating core Ray primitives."""

import ray
import collections


@ray.remote
def tokenize(text: str) -> dict[str, int]:
    """Count word frequencies in a text chunk."""
    counts: dict[str, int] = {}
    for word in text.lower().split():
        word = word.strip(".,!?;:'\"()[]")
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts


@ray.remote
class WordCounter:
    """Actor that accumulates word counts across chunks."""

    def __init__(self):
        self.total: dict[str, int] = {}

    def merge(self, counts: dict[str, int]):
        for word, count in counts.items():
            self.total[word] = self.total.get(word, 0) + count

    def top_k(self, k: int = 10) -> list[tuple[str, int]]:
        return sorted(self.total.items(), key=lambda x: -x[1])[:k]

    def total_words(self) -> int:
        return sum(self.total.values())


def main():
    ray.init()

    texts = [
        "Ray is a unified framework for scaling AI and Python applications.",
        "It provides core primitives like tasks actors and objects for distributed computing.",
        "Ray Train handles distributed training across multiple GPUs and nodes.",
        "Ray Data provides distributed data processing pipelines.",
        "Ray Serve enables scalable model serving with dynamic batching.",
        "Ray Tune offers hyperparameter tuning with various search algorithms.",
        "The Ray ecosystem makes it easy to scale from laptop to cluster.",
        "Ray integrates with PyTorch TensorFlow JAX and HuggingFace Transformers.",
    ]

    # Map: parallel tokenization
    count_refs = [tokenize.remote(text) for text in texts]
    all_counts = ray.get(count_refs)

    # Reduce: merge via actor
    counter = WordCounter.remote()
    for counts in all_counts:
        counter.merge.remote(counts)

    total = ray.get(counter.total_words.remote())
    top = ray.get(counter.top_k.remote(10))

    print(f"\nProcessed {len(texts)} text chunks, {total} total words")
    print("Top 10 words:")
    for word, count in top:
        print(f"  {word}: {count}")

    ray.shutdown()


if __name__ == "__main__":
    main()
