import requests
from zipfile import ZipFile
from io import BytesIO

print("Downloading Tiny Imagenet")
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
response = requests.get(url, stream=True)
print(f"Response status: {response}")

if response.status_code == 200:
    print("Extracting data")
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall("TinyImagenet")
    print("Download and extraction complete!")
else:
    print("Failed to download the dataset.")
