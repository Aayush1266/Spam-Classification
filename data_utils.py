import os
import zipfile
import requests
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Spam-Dataset
# Dataset download URLs & paths
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = "sms_spam_collection.zip"
EXTRACTED_PATH = "sms_spam_collection"
DATA_FILE_PATH = Path(EXTRACTED_PATH) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(
    url=URL, zip_path=ZIP_PATH, extracted_path=EXTRACTED_PATH, data_file_path=DATA_FILE_PATH
):
    if data_file_path.exists():
        return
    response = requests.get(url)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        out_file.write(response.content)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    os.rename(Path(extracted_path) / "SMSSpamCollection", data_file_path)


def create_balanced_dataset(df):
    # separate spam and ham data
    spam_df = df[df.Label == "spam"]
    ham_df = df[df.Label == "ham"]

    # Count
    num_spam = len(spam_df)
    num_ham = len(ham_df)

    # sample according to dataset with lower count
    if num_spam < num_ham:
        ham_df = ham_df.sample(num_spam, random_state=123)
    else:
        spam_df = spam_df.sample(num_ham, random_state=123)

    balanced_df = pd.concat([spam_df, ham_df]).sample(
        frac=1, random_state=123
        ).reset_index(drop=True)
    balanced_df.Label = balanced_df.Label.map({"ham": 0, "spam": 1})

    return balanced_df



def random_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)
    return df[:train_end], df[train_end:val_end], df[val_end:]


class SpamDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=None, pad_token_id=50256):
        self.texts = df.Text.tolist()
        self.labels = df.Label.tolist()
        self.encoded_texts = [
            tokenizer.encode(t) for t in self.texts
        ]

        self.max_length = max_length or max(len(enc) for enc in self.encoded_texts)

        # Pad or truncate
        self.encoded_texts = [
            enc[: self.max_length] + [pad_token_id] * max(0, self.max_length - len(enc)) for enc in self.encoded_texts
        ]

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_texts[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def load_data(tokenizer, batch_size=8, train_frac=0.7, val_frac=0.1):
    # Download & preprocess
    download_and_unzip_spam_data()
    df = pd.read_csv(DATA_FILE_PATH, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    train_df, val_df, test_df = random_split(balanced_df, train_frac, val_frac)

    # Save to CSV
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("validation.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    # decide max length from all 
    all_texts = pd.concat([train_df.Text, val_df.Text, test_df.Text])
    max_length = max(len(tokenizer.encode(t)) for t in all_texts)

    # Datasets
    train_data = SpamDataset(train_df, tokenizer, max_length=max_length)
    val_data = SpamDataset(val_df, tokenizer, max_length=train_data.max_length)
    test_data = SpamDataset(test_df, tokenizer, max_length=train_data.max_length)

    # print(train_data.max_length)

    # Loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_data.max_length


# =========================================================================
# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")

# train_loader, val_loader, test_loader, max_length = load_data(tokenizer)

# print(len(train_loader))
# print(len(val_loader))
# print(len(test_loader))