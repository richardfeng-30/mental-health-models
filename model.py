import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

# Step 1: Load and Explore the Dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path, encoding='latin-1', header=None)

# Check the label distribution
print(data[0].value_counts())

# Assuming binary classification (0 and 4), let's map label 4 to 1
data[0] = data[0].map({0: 0, 4: 1})

# Split the data
X = data.iloc[:, -1].tolist()
y = data.iloc[:, 0].tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)


# Custom Dataset Class
class MentalHealthDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Create dataset objects
train_dataset = MentalHealthDataset(train_encodings, y_train)
val_dataset = MentalHealthDataset(val_encodings, y_val)

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

# Step 6: Inference with the Trained Model
# Load the saved model and tokenizer
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')

# Set the model to evaluation mode
model.eval()

# Example input text
text = "I'm feeling very depressed and anxious today."

# Tokenize the input text
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)

# Move tensors to GPU if available
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted label
prediction = torch.argmax(outputs.logits, dim=1).item()

# Map the prediction to the corresponding class (assuming 0: Negative, 1: Positive)
label_map = {0: 'Negative', 1: 'Positive'}
predicted_label = label_map[prediction]

print(f"Predicted label: {predicted_label}")

# Testing the Model with Multiple Inputs
test_texts = [
    "I'm feeling very happy and excited today!",
    "I'm so sad and lonely.",
    "I don't know what to do with my life."
]

# Make predictions for each text
for text in test_texts:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = label_map[prediction]

    print(f"Text: {text}\nPredicted label: {predicted_label}\n")
