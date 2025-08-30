import boto3
import glob
import re

# Need to set up AWS credentials in ~/.aws/credentials first

client = boto3.client("s3")

files = glob.glob("/home/ubuntu/data/artifact/dim192/*")
for file in files:
    dst = re.sub(r'/home/ubuntu', '', file)
    print(dst)
    client.upload_file(file, "tiptoe-artifact-eval", dst)

client.upload_file("/home/ubuntu/data/artifact/dim192/index.faiss", "tiptoe-artifact-eval", "/data/artifact/dim192/index.faiss")
client.upload_file("/home/ubuntu/data/artifact/dim192/pca_192.npy", "tiptoe-artifact-eval", "/data/artifact/dim192/pca_192.npy")
