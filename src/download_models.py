import os
import urllib
from urllib.request import urlretrieve


def download_language_model(source, target):
    model = f"opus-mt-{source}-{target}"
    print(">>>Downloading data for %s to %s model..." % (source, target))
    os.makedirs(os.path.join("data", model))
    for f in FILENAMES:
        try:
            print(os.path.join(HUGGINGFACE_S3_BASE_URL, model, f))
            urlretrieve(
                "/".join([HUGGINGFACE_S3_BASE_URL, model, f]),
                os.path.join(MODEL_PATH, model, f),
            )
            print("Download complete!")
        except urllib.error.HTTPError:
            print("Error retrieving model from url. Please confirm model exists.")
            os.rmdir(os.path.join("data", model))
            break


if __name__ == "__main__":
    HUGGINGFACE_S3_BASE_URL="https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP"
    FILENAMES = ["config.json","pytorch_model.bin","source.spm","target.spm","tokenizer_config.json","vocab.json"]
    MODEL_PATH = "./data"
    download_language_model('en','hi')