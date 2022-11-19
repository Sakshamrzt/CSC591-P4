import os.path
import numpy as np

from scipy.special import softmax

from tvm.contrib.download import download_testdata

# Download a list of labels
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]
# print(labels)
labels=[0,1,2,3,4,5,6,7,8,9]
output_file = "predictions.npz"

# Open the output and read the output tensor
if os.path.exists(output_file):
    with np.load(output_file) as data:
        print(data)
        scores = softmax(data["output_0"])
        scores = np.squeeze(scores)
        print(scores)
        ranks = np.argsort(scores)[::-1]
        print(scores, ranks)

        for rank in ranks[0:5]:
            print("class='%s' with probability=%f" % (labels[rank], scores[rank]))