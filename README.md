# Hidden Markov Model (HMM) Implementation with Four Algorithms for POS Tagging

## Overview

This project implements a Hidden Markov Model (HMM) using four different algorithms for sequence tagging and evaluation. The algorithms are:

1. **Viterbi Algorithm**
2. **Beam Search**
3. **Greedy Search**
4. **Posterior Decoding**

The implementation is designed to train and test the HMM on datasets for multiple languages, evaluate the results using precision, recall, F1 score, and confusion matrix, and save the output for each algorithm.

---

## Features

- **Multi-language support:** Includes pre-trained datasets for English, Hindi, Spanish, and Sanskrit.
- **Four algorithms:** Implements different decoding strategies for comparative evaluation.
- **Custom datasets:** Supports training with user-provided datasets.
- **Result storage:** Saves evaluation metrics and confusion matrices for each algorithm in language-specific folders.

---

## Usage

### Running the Model

Run the following command based on your operating system:

- **Windows:**
  ```bash
  python main.py
  ```
- **UNIX-based systems:**
  ```bash
  python3 main.py
  ```

### Input

1. The program prompts you to:
   - **Enter the language** for training and testing (e.g., English, Hindi, etc.).
   - **Provide the path to the training data file**.
     - If no file is provided or the file doesn't exist, the code downloads the default dataset for supported languages from the web.
     - Enter `NA` to use the default dataset.

### Output

- Results for each algorithm are saved in the `Results` folder, under subdirectories for each language.
- The result file, named as `<algorithm_name>_results.txt`, contains:
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

### Example Output Structure

```
Results/
└── English/
    ├── viterbi_results.txt
    ├── beam_search_results.txt
    ├── greedy_search_results.txt
    └── posterior_decoding_results.txt
```

---

## Algorithms

### 1. Viterbi Algorithm

- A dynamic programming-based decoding algorithm.
- Finds the most likely sequence of hidden states.

### 2. Beam Search

- Extends the Viterbi Algorithm with a beam width to limit possible state paths.
- Trades accuracy for efficiency with a configurable beam width (default = 3).

### 3. Greedy Search

- Makes locally optimal choices at each step.
- A simple but less accurate algorithm.

### 4. Posterior Decoding

- Calculates posterior probabilities for all states in each position.
- Decodes based on the most likely state for each position independently.

---

## Evaluation Metrics

The program evaluates model performance using:

1. **Precision**
2. **Recall**
3. **F1 Score**
4. **Confusion Matrix**

---

## Customization

### Adding Training Data

- Place your dataset file in the `Dataset` directory.
- Enter the file path when prompted during program execution.

### Changing Beam Width

- Modify the `beam_width` parameter in the `test_beam_search` function in `testing.py`.

---

## Dependencies

The project uses the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`

---

## Notes

- Default datasets are sourced from the [Universal Dependencies project](https://universaldependencies.org/).
- Ensure a stable internet connection for downloading datasets if not already available.
