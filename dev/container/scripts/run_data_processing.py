"""Process FineWeb dataset with Ray Data."""

import ray
import ray.data
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "/data/outputs/fineweb-processed"
FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_SUBSET = "sample-10BT"

# Processing parameters
MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH = 50_000
MAX_TOKEN_LENGTH = 2048


def load_fineweb() -> ray.data.Dataset:
    """Load FineWeb from HuggingFace cache via datasets library."""
    from datasets import load_dataset

    hf_ds = load_dataset(
        FINEWEB_DATASET,
        name=FINEWEB_SUBSET,
        split="train",
        streaming=True,
    )
    # Take a manageable subset for demonstration
    rows = []
    for i, example in enumerate(hf_ds):
        if i >= 10_000:
            break
        rows.append({"text": example["text"], "url": example.get("url", "")})

    return ray.data.from_items(rows)


class TextCleaner:
    """Filter and clean text documents."""

    def __call__(self, batch: dict) -> dict:
        texts = batch["text"]
        urls = batch["url"]
        clean_texts = []
        clean_urls = []
        for text, url in zip(texts, urls):
            text = text.strip()
            if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
                continue
            # Basic dedup heuristic: skip if too much repetition
            words = text.split()
            if len(words) > 0 and len(set(words)) / len(words) < 0.3:
                continue
            clean_texts.append(text)
            clean_urls.append(url)
        return {"text": clean_texts, "url": clean_urls}


class Tokenizer:
    """Tokenize text and compute token counts."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __call__(self, batch: dict) -> dict:
        texts = [str(t) for t in batch["text"]]
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding=False,
        )
        return {
            "text": texts,
            "url": [str(u) for u in batch["url"]],
            "token_count": [len(ids) for ids in encodings["input_ids"]],
            "char_count": [len(t) for t in texts],
        }


def main():
    ray.init()

    print("Loading FineWeb dataset...")
    ds = load_fineweb()
    initial_count = ds.count()
    print(f"Loaded {initial_count} documents")

    print("Cleaning and filtering...")
    ds = ds.map_batches(TextCleaner, batch_size=256, concurrency=2)

    print("Tokenizing...")
    ds = ds.map_batches(Tokenizer, batch_size=256, concurrency=2)

    # Compute statistics before writing
    count = ds.count()
    print(f"\nProcessing statistics:")
    print(f"  Documents after filtering: {count}")

    print(f"\nWriting processed data to {OUTPUT_DIR}...")
    ds.write_parquet(OUTPUT_DIR)

    print("Data processing complete.")
    ray.shutdown()


if __name__ == "__main__":
    main()
