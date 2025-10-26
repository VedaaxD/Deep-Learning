#super concise, reusable SKELETON for any PyTorch dataset, no matter if it’s text, images, or tabular data:

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, labels_source, transform=None):
        self.data_source = data_source      # raw inputs (list, DataFrame, etc.)
        self.labels_source = labels_source  # raw labels/targets
        self.transform = transform          # optional preprocessing

    def __len__(self):
        return len(self.data_source)        # total number of samples

    def __getitem__(self, index):
        # 1. Get raw input and label
        x = self.data_source[index]
        y = self.labels_source[index]

        # 2. Optional preprocessing / transform
        if self.transform:
            x = self.transform(x)

        # 3. Convert to tensors (if not already)
        x = torch.tensor(x)
        y = torch.tensor(y)

        # 4. Return sample
        return x, y
# How to use it
# For text:
#
# data_source → list of questions
#
# labels_source → list of answers
#
# transform → function to tokenize and convert to indices
#
# For images:
#
# data_source → list of image file paths
#
# labels_source → class labels
#
# transform → torchvision.transforms for preprocessing
#
# For tabular data:
#
# data_source → NumPy array or DataFrame values
#
# labels_source → targets column
#
# transform → optional normalization/scaling
#
# 💡 Key takeaway:
#
# __getitem__ always returns one sample in tensor format, ready for your model.
#
# Preprocessing can be done inside __getitem__ or via a transform.