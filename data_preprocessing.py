import pandas as pd
import os
import re

DATA_FOLDER = "database"


def clean_title(title):
    """Advanced cleaning"""
    if not isinstance(title, str):
        return None

    title = title.strip().lower()

    # remove extra spaces
    title = re.sub(r"\s+", " ", title)

    # remove unwanted characters (keep letters + spaces)
    title = re.sub(r"[^a-zA-Z\s]", "", title)

    # remove very short titles
    if len(title) < 4:
        return None

    return title


def extract_titles_from_file(filepath):
    try:
        df = pd.read_csv(filepath)

        # normalize column names
        df.columns = [col.strip().lower() for col in df.columns]

        if "title" not in df.columns:
            print(f"❌ No title column in {filepath}")
            return []

        titles = df["title"].dropna()

        cleaned_titles = []

        for t in titles:
            clean = clean_title(t)
            if clean:
                cleaned_titles.append(clean)

        return cleaned_titles

    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return []


def load_all_titles():
    all_titles = []

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".csv"):
            path = os.path.join(DATA_FOLDER, file)

            print(f"Processing: {file}")

            titles = extract_titles_from_file(path)
            all_titles.extend(titles)

    print(f"\nBefore deduplication: {len(all_titles)}")

    # remove duplicates
    unique_titles = list(set(all_titles))

    print(f"After deduplication: {len(unique_titles)}")

    return unique_titles


def save_clean_dataset(titles):
    df = pd.DataFrame({"title": titles})
    df.to_csv("database/cleaned_titles.csv", index=False)
    print("\n✅ Clean dataset saved: database/cleaned_titles.csv")


if __name__ == "__main__":
    titles = load_all_titles()

    save_clean_dataset(titles)

    print("\n🔍 Sample titles:")
    for t in titles[:10]:
        print(t)