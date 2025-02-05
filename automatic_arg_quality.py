import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from scipy.stats import pearsonr, spearmanr

def create_pairs_and_labels(tsv_file):
  try:
    df = pd.read_csv(tsv_file, sep='\t')
    pairs = list(zip(df['a1'], df['a2']))
    labels = df['label'].tolist()
    return pairs, labels
  except FileNotFoundError:
    print(f"Error: File not found: {tsv_file}")
    return [], []
  except Exception as e:
    print(f"An error occurred: {e}")
    return [], []
  
def create_args_and_labels(tsv_file):
  try:
    df = pd.read_csv(tsv_file, sep='\t')
    args = df['argument'].tolist()
    labels = df['rank'].tolist()
    return args, labels
  except FileNotFoundError:
    print(f"Error: File not found: {tsv_file}")
    return [], []
  except Exception as e:
    print(f"An error occurred: {e}")
    return [], []

def create_args_and_labels_test(csv_file):
  try:
    df = pd.read_csv(csv_file)
    args = df['argument'].tolist()
    labels = df['WA'].tolist()
    return args, labels
  except FileNotFoundError:
    print(f"Error: File not found: {csv_file}")
    return [], []
  except Exception as e:
    print(f"An error occurred: {e}")
    return [], []

# Step 1: Load pre-trained BERT and fine-tune it for binary classification
class ArgClassifier(nn.Module):
    def __init__(self):
        super(ArgClassifier, self).__init__()
        # Input: Dataloader passes tokenized [CLS]argA[SEP]argB type embeddings
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.classifier = nn.Linear(768, 2)  # Output layer for binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token representation
        logits = self.classifier(cls_output)
        return logits

# Custom Dataset for Argument Pair Classification (for fine-tuning the BERT embeddings)
class ArgPairDataset(Dataset):
    def __init__(self, arg_pairs, labels, tokenizer, max_length=512):
        self.arg_pairs = arg_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.arg_pairs)

    def __getitem__(self, idx):
        arg_a, arg_b = self.arg_pairs[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            f"[CLS]{arg_a}[SEP]{arg_b}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# Fine-tuning the ArgClassifier
# Input: Dataloader passes (not fine tuned) BERT-embeddings 
def train_arg_classifier(model, dataloader, epochs=3, lr=2e-5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}")

    return model

# Save the fine-tuned model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Step 2: Define the Arg-Ranker model
class ArgRanker(nn.Module):
    def __init__(self):
        super(ArgRanker, self).__init__()
        self.hidden_layer = nn.Linear(3072, 300)  # First layer
        self.output_layer = nn.Linear(300, 1)  # Output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden_output = self.relu(self.hidden_layer(x))
        output = self.sigmoid(self.output_layer(hidden_output))
        return output

# Step 3: Fine-tuning and embedding extraction
class FineTunedBertForRanking:
    def __init__(self, arg_classifier_checkpoint):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = ArgClassifier()
        self.model.load_state_dict(torch.load(arg_classifier_checkpoint, weights_only=True))
        self.model.to("cuda")
        self.model.eval()

    def get_argument_embedding(self, argument):
        # Tokenize single argument
        inputs = self.tokenizer(
            f"[CLS]{argument}[SEP]",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()} 
        outputs=[]
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
        # Concatenate last 4 layers
        last_hidden_states = outputs.hidden_states[-4:]  # Last 4 layers
        concatenated = torch.cat(last_hidden_states, dim=-1)  # Shape: (batch_size, seq_len, 3072)
        cls_embedding = concatenated[:, 0, :]  # Extract [CLS] embedding
        return cls_embedding

# Custom Dataset for Argument Ranking
class ArgRankingDataset(Dataset):
    def __init__(self, arguments, scores, tokenizer, max_length=512):
        self.arguments = arguments
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.arguments)

    def __getitem__(self, idx):
        argument = self.arguments[idx]
        score = self.scores[idx]
        encoded = self.tokenizer(
            f"[CLS]{argument}[SEP]",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return input_ids, attention_mask, torch.tensor(score, dtype=torch.float)

# Training loop for Arg-Ranker
def train_arg_ranker(arg_ranker, dataloader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(arg_ranker.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arg_ranker.to(device)
    arg_ranker.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for input_ids, attention_mask, scores in dataloader:
            input_ids, attention_mask, scores = input_ids.to(device), attention_mask.to(device), scores.to(device)

            optimizer.zero_grad()
            embeddings = fine_tuned_bert.get_argument_embedding(input_ids).to(device)
            outputs = arg_ranker(embeddings)
            loss = criterion(outputs.squeeze(), scores)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        arg_ranker_checkpoint = f"arg_ranker_{epoch}_fin.pth"
        save_model(arg_ranker, arg_ranker_checkpoint)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
    
    return arg_ranker

# Example usage
if __name__ == "__main__":
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tsv_path = "/workspace/ashish98/ashish_ug/kannan/argument_quality/automatic_arg_quality_implementation/arg_pairs.tsv"  
    arg_pairs, labels = create_pairs_and_labels(tsv_path)

    dataset = ArgPairDataset(arg_pairs, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Fine-tune ArgClassifier
    arg_classifier = ArgClassifier()
    fine_tuned_classifier = train_arg_classifier(arg_classifier, dataloader, epochs=3, lr=2e-5)

    # Save the fine-tuned model
    classifier_checkpoint = "fine_tuned_arg_classifier_2.pth"
    save_model(fine_tuned_classifier, classifier_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load fine-tuned Arg-Classifier for ranking
    fine_tuned_bert = FineTunedBertForRanking(classifier_checkpoint)

    tsv_path1 = "/workspace/ashish98/ashish_ug/kannan/argument_quality/automatic_arg_quality_implementation/arg.tsv"  
    arguments, scores = create_args_and_labels(tsv_path1)

    # Prepare dataset and dataloader for ranking
    ranking_dataset = ArgRankingDataset(arguments, scores, tokenizer)
    ranking_dataloader = DataLoader(ranking_dataset, batch_size=1, shuffle=True)

    # Train Arg-Ranker
    arg_ranker = ArgRanker()
    final_arg_ranker=train_arg_ranker(arg_ranker, ranking_dataloader, epochs=10, lr=0.001)
    final_arg_ranker_checkpoint = "arg_ranker_9_fin.pth"
    # save_model(final_arg_ranker, final_arg_ranker_checkpoint)

    final_arg_ranker = ArgRanker()
    final_arg_ranker.load_state_dict(torch.load(final_arg_ranker_checkpoint, weights_only=True))
    final_arg_ranker.to(device)
    final_arg_ranker.eval()


    # Prepare data for prediction
    csv_path = "/workspace/ashish98/ashish_ug/kannan/argument_quality/arg_quality_rank_30k.csv"  # Replace with the path to your argument-score tsv file
    arguments, actual_scores = create_args_and_labels_test(csv_path)
    print(len(arguments))
    dataset = ArgRankingDataset(arguments, actual_scores, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Disable shuffling for evaluation

    # Calculate loss (Mean Squared Error)
    criterion = nn.MSELoss()
    total_loss = 0

    predictions = []
    actual_scores_list = []

    with torch.no_grad():
        for input_ids, attention_mask, scores in dataloader:
            input_ids, attention_mask, scores = input_ids.to(device), attention_mask.to(device), scores.to(device)
            embeddings = fine_tuned_bert.get_argument_embedding(input_ids).to(device)

            outputs = final_arg_ranker(embeddings)
            loss = criterion(outputs.squeeze(), scores)
            print("model output:")
            print(outputs)
            print("actual score:")
            print(scores)

            predictions.append(outputs.squeeze().cpu().item())
            actual_scores_list.append(scores.cpu().item())
            total_loss += loss.item()

    # Save predictions and actual scores to a CSV file
    results_df = pd.DataFrame({
        "Predictions": predictions,
        "Actual Scores": actual_scores_list
    })

    results_file = "predictions_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Predictions and actual scores saved to {results_file}")

    # Calculate and print average loss
    average_loss = total_loss / len(dataloader)
    print(f"Average Loss on the IBM 30k dataset: {average_loss:.4f}")

    # Calculate Pearson's and Spearman's correlation
    pearson_corr, _ = pearsonr(predictions, actual_scores_list)
    spearman_corr, _ = spearmanr(predictions, actual_scores_list)
    print(f"Pearson's Correlation: {pearson_corr:.4f}")
    print(f"Spearman's Correlation: {spearman_corr:.4f}")
