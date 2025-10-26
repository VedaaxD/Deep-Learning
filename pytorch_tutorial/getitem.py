#REFRESHER
#__getitem__ method
def __getitem__(self, index):
    # 1️⃣ Get the raw data for this index
    raw_input = self.data_source[index]       # e.g., row in DataFrame, or file in a list
    raw_label = self.labels_source[index]     # e.g., target/output

    # 2️⃣ Preprocess the input (convert to numeric, normalize, tokenize, etc.)
    processed_input = preprocess_input(raw_input)

    # 3️⃣ Preprocess the label (if needed)
    processed_label = preprocess_label(raw_label)

    # 4️⃣ Convert to PyTorch tensors
    input_tensor = torch.tensor(processed_input)
    label_tensor = torch.tensor(processed_label)

    # 5️⃣ Return as a tuple (input, label)
    return input_tensor, label_tensor
#