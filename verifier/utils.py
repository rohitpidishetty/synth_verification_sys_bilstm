import sys
sys.path.append("./model/torch") 

import re
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import torch
from verifier.nnModel import BiLSTM_Attention_Model


import os
from azure.storage.blob import BlobServiceClient
from utils.download import download_model

def download_blob_folder(container_name, blob_prefix, local_path):
    conn_str = os.getenv("DefaultEndpointsProtocol=https;AccountName=synthauditor;AccountKey=tVbibIqXWK4VYrRzZ0GHgsSJC2AQ8RLBaaDotQG/TLvRLf+WF74xLmPkgD6MwPNb6i1L8FFdOHf3+AStzczhwg==;EndpointSuffix=core.windows.net")
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service_client.get_container_client(container_name)

    os.makedirs(local_path, exist_ok=True)

    blobs = container_client.list_blobs(name_starts_with=blob_prefix)
    for blob in blobs:
        blob_client = container_client.get_blob_client(blob.name)
        rel_path = os.path.relpath(blob.name, blob_prefix)
        local_file_path = os.path.join(local_path, rel_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        print(f"Downloaded {blob.name} -> {local_file_path}")

st_local_path = "./model/sentence-transformer-all-mpnet-base-v2"
download_blob_folder("model", "sentence-transformer-all-mpnet-base-v2/", st_local_path)

stopwords = [
    # Standard stopwords
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "arent",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "cant",
    "cannot",
    "could",
    "couldnt",
    "did",
    "didnt",
    "do",
    "does",
    "doesnt",
    "doing",
    "dont",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadnt",
    "has",
    "hasnt",
    "have",
    "havent",
    "having",
    "he",
    "hed",
    "hell",
    "hes",
    "her",
    "here",
    "heres",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "hows",
    "i",
    "id",
    "ill",
    "im",
    "ive",
    "if",
    "in",
    "into",
    "is",
    "isnt",
    "it",
    "its",
    "itself",
    "lets",
    "me",
    "more",
    "most",
    "mustnt",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "shed",
    "shell",
    "shes",
    "should",
    "shouldnt",
    "so",
    "some",
    "such",
    "than",
    "that",
    "thats",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "theres",
    "these",
    "they",
    "theyd",
    "theyll",
    "theyre",
    "theyve",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasnt",
    "we",
    "wed",
    "well",
    "were",
    "weve",
    "were",
    "werent",
    "what",
    "whats",
    "when",
    "whens",
    "where",
    "wheres",
    "which",
    "while",
    "who",
    "whos",
    "whom",
    "why",
    "whys",
    "with",
    "wont",
    "would",
    "wouldnt",
    "you",
    "youd",
    "youll",
    "youre",
    "youve",
    "your",
    "yours",
    "yourself",
    "yourselves",
    # Social media filler words
    "rt",
    "via",
    "lol",
    "lmao",
    "omg",
    "idk",
    "tbh",
    "btw",
    "pls",
    "plz",
    "u",
    "ur",
    "r",
    "imho",
    "irl",
    "smh",
    "fyi",
    "yea",
    "yeah",
    "yup",
    "nope",
    "okay",
    "ok",
    "k",
    # Noise words
    "breaking",
    "update",
    "alert",
    "exclusive",
    "viral",
    "share",
    "repost",
    "read",
    "watch",
    "click",
    "follow",
    "true",
    "false",
    "real",
    "fake",
    "hoax",
    "scam",
    # Very frequent low-signal verbs
    "say",
    "says",
    "said",
    "tell",
    "told",
    "think",
    "thought",
    "know",
    "known",
    "report",
    "reported",
    "claim",
    "claimed",
    "claims",
    "show",
    "shown",
    "shows",
    "make",
    "makes",
    "made",
    "see",
    "seen",
    "look",
    "looks",
    # Generic nouns that almost never contribute to classification
    "people",
    "person",
    "man",
    "woman",
    "guy",
    "guys",
    "thing",
    "stuff",
    "someone",
    "everyone",
    "anyone",
    # Misinformation bait
    "wow",
    "shocking",
    "unbelievable",
    "insane",
    "must",
    "watch",
    "truth",
    "facts",
    "omfg",
    "literally",
]

lem = WordNetLemmatizer()
model = SentenceTransformer(st_local_path)


download_model()


path = "./model/torch"

download_blob_folder("model", "torch/", path)


model_lstm = BiLSTM_Attention_Model(input_dim=768, hidden_dim=256, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_lstm.load_state_dict(
    torch.load("./model/yaari-synth-auditor-model-v1.pth", map_location=device)
)
model_lstm.to(device)

model_lstm.eval()


def cleanse(word):
    buffer = word.split()
    stream = ""
    for token in buffer:
        clean = lem.lemmatize(re.sub(r"[^a-zA-Z0-9]", "", token).lower()).strip()
        if clean not in stopwords:
            stream += clean + " "
    return stream



def auditor(data):
  emb = model.encode(data)
  X_single = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)  
  with torch.no_grad():
      outputs, _ = model_lstm(X_single.to(device))
      pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1).item()
  return pred_class






