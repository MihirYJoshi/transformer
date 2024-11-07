import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os

from tokenizer import SimpleTokenizer
from utilities import Utilities

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import sys
import nltk
#nltk.download('punkt')

from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Encoder, FeedForward, Decoder, EncoderClassifier

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(
        padded_sequences,
        (0, max(0, block_size - padded_sequences.shape[1])),
        "constant",
        0,
    )
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
   

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(
        padded_sequences,
        (0, max(0, block_size - padded_sequences.shape[1])),
        "constant",
        0,
    )
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = 100 * total_correct / total_samples
        classifier.train()
        return accuracy
    
def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(
            X, Y
        )  # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters:
            break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def main():
    type = str(sys.argv[1])

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)


    if type == "part1":
        # Initialize models
        combined_model = EncoderClassifier(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size, n_hidden).to(device)


        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(list(combined_model.parameters()), lr=learning_rate)

        # for the classification  task, you will train for a fixed number of epochs like this:
        
        for epoch in range(epochs_CLS):
            total_loss = 0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                outputs, _ = combined_model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            train_accuracy = compute_classifier_accuracy(combined_model, train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(combined_model, test_CLS_loader)
            
            print(f"Epoch {epoch+1}/{epochs_CLS}, Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
        
        # Final evaluation on test set
        final_test_accuracy = compute_classifier_accuracy(combined_model, test_CLS_loader)
        print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")

        # Print number of parameters
        encoder_params = sum(p.numel() for p in combined_model.parameters())
        print(f"Num of parameters in EncoderClassifier: {encoder_params}")

        from utilities import Utilities
        utilities = Utilities(tokenizer, combined_model)

        sentences = [
            "And Democrats, we must also admit that fulfilling America's promise will require more than just money."
        ]

        for sentence in sentences:
            utilities.sanity_check(sentence, block_size)

    
    
    elif type == "part2":

        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        

        inputfile = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            hbush = f.read()
        hbush_LM_dataset = LanguageModelingDataset(tokenizer, hbush,  block_size)
        hbush_LM_loader = DataLoader(hbush_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            wbush = f.read()
        wbush_LM_dataset = LanguageModelingDataset(tokenizer, wbush,  block_size)
        wbush_LM_loader = DataLoader(wbush_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_obama.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            obama = f.read()
        obama_LM_dataset = LanguageModelingDataset(tokenizer, obama,  block_size)
        obama_LM_loader = DataLoader(obama_LM_dataset, batch_size=batch_size, shuffle=True)



        decoder = Decoder(tokenizer.vocab_size, n_embd, n_layer, n_head).to(device)

        optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

        total_loss = 0
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            # Forward pass through decoder
            loss, _ = decoder(xb, yb)

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                avg_loss = total_loss / (i + 1)
                perplexity = compute_perplexity(decoder, train_LM_loader)
                print(f"Iteration {i + 1}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")


        obama_perplexity = compute_perplexity(decoderLMmodel=decoder,
                                                data_loader=obama_LM_loader)
        print(f"Obama Test Set Perplexity: {obama_perplexity:.4f}")

        # Evaluate on W. Bush test set
        wbush_perplexity = compute_perplexity(decoderLMmodel=decoder,
                                            data_loader=wbush_LM_loader)
        print(f"W. Bush Test Set Perplexity: {wbush_perplexity:.4f}")

        # Evaluate on H. Bush test set
        ghbush_perplexity = compute_perplexity(decoderLMmodel=decoder,
                                            data_loader=hbush_LM_loader)
        print(f"H. Bush Test Set Perplexity: {ghbush_perplexity:.4f}")

        # Print number of parameters
        decoder_params = sum(p.numel() for p in decoder.parameters())
        print(f"Num of parameters in Decoder: {decoder_params}")

        from utilities import Utilities
        utilities = Utilities(tokenizer, decoder)

        sentences = [
            "And Democrats, we must also admit that fulfilling America's promise will require more than just money."
        ]

        for sentence in sentences:
            utilities.sanity_check(sentence, block_size)


    else:
        print("Incorrect type. Valid Types are 'part1'or 'part2'")

    



if __name__ == "__main__":
    main()
