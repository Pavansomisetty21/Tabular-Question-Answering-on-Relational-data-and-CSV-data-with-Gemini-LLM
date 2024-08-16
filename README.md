# Tabular-Question-Answering-on-Relational-data-and-CSV-data-with-Gemini-LLM
In this we explore into a Question Answering task on structured relational  data (Tables) and CSV data
also we perform Table Question Answering using tapas-base-finetuned-wtq
# Tabular Question Answering with TAPAS

## Overview

This repository provides a fine-tuned version of the TAPAS model, specifically `tapas-base-finetuned-wtq`, for tabular question answering tasks. TAPAS is a pre-trained language model designed to handle questions about tabular data, leveraging its ability to reason over structured tables.
TAPAS is a BERT-like transformers model pretrained on a large corpus of English data from Wikipedia in a self-supervised fashion. This means it was pretrained on the raw tables and associated texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:

Masked language modeling (MLM): taking a (flattened) table and associated context, the model randomly masks 15% of the words in the input, then runs the entire (partially masked) sequence through the model. The model then has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of a table and associated text.

Intermediate pre-training: to encourage numerical reasoning on tables, the authors additionally pre-trained the model by creating a balanced dataset of millions of syntactically created training examples. Here, the model must predict (classify) whether a sentence is supported or refuted by the contents of a table. The training examples are created based on synthetic as well as counterfactual statements.
This way, the model learns an inner representation of the English language used in tables and associated texts, which can then be used to extract features useful for downstream tasks such as answering questions about a table, or determining whether a sentence is entailed or refuted by the contents of a table. Fine-tuning is done by adding a cell selection head and aggregation head on top of the pre-trained model, and then jointly train these randomly initialized classification heads with the base model on SQa, WikiSQL and finally WTQ

The `tapas-base-finetuned-wtq` model has been fine-tuned on the WikiTableQuestions (WTQ) dataset, allowing it to effectively interpret and answer questions based on the content of tables extracted from Wikipedia.

![Image](https://1.bp.blogspot.com/-SOS5yrSg0lw/XqsC0RyAXiI/AAAAAAAAF3g/BcOoE84UY64QtwoZC06YEe_6SblvxMncgCLcBGAsYHQ/s1600/image1.png)


![image1](https://1.bp.blogspot.com/-Rh7FdX9e4x4/XqsCvW8drRI/AAAAAAAAF3c/Lt_uYMCGKeUR2wmiB2Qd2yOF4hNDvoTBwCLcBGAsYHQ/s1600/image2.png)

for more details we can see  [Understanding tables with intermediate pre-training](https://www.aclweb.org/anthology/2020.findings-emnlp.27/)


## Features

- **Tabular Data Handling:** Efficiently processes and interprets tabular data for answering questions.
- **Contextual Understanding:** Combines table content with question context to provide accurate answers.
- **Complex Query Handling:** Capable of dealing with queries requiring aggregations, comparisons, and filtering.

## Setup

To get started with the `tapas-base-finetuned-wtq` model, follow these steps:

### Prerequisites

Ensure you have Python 3.6+ and the necessary libraries installed. You can use the following command to install the required libraries:

```bash
pip install transformers
pip install torch
```
## Usage

You can use the tapas-base-finetuned-wtq model for tabular question answering as follows:

code
```python
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Step 1: Prepare your table as a pandas DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": ["24", "19", "22"],
    "City": ["New York", "Los Angeles", "Chicago"]
}
table = pd.DataFrame.from_dict(data)

# Step 2: Prepare your question
question = "What is the age of Bob?"

# Step 3: Tokenize the inputs
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
inputs = tokenizer(table=table, queries=question, padding="max_length", return_tensors="pt")

# Step 4: Run the model
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
outputs = model(**inputs)

# Step 5: Post-process the output to get the answer
predicted_answer_coordinates, predicted_aggregation_index = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
)

# Debug: Print the predicted answer coordinates
print(f"Predicted answer coordinates: {predicted_answer_coordinates}")

# Check if coordinates exist and are valid
if predicted_answer_coordinates and predicted_answer_coordinates[0]:
    try:
        # Unpack the coordinates from the nested list
        row_index, col_index = predicted_answer_coordinates[0][0]
        # If the predicted answer is a cell, extract the cell's value
        answer = table.iat[row_index, col_index]
    except IndexError:
        answer = "Invalid answer coordinates: out of range"
else:
    answer = "No answer found"

print(f"Question: {question}")
print(f"Answer: {answer}")

```
for the above code we may get the output in the form of as follows
## output
```
Predicted answer coordinates: [[(1, 1)]]
Question: What is the age of Bob?
Answer: 19
```


now we explore in the another LLM for Tabular question Answering task calles `GEMINI`

# Tabular Question Answering using Google GEMINI LLM


## Overview

This project demonstrates how to use the Google Gemini API to perform tabular question answering. The script leverages the `tapas-base-finetuned-wtq` model to answer questions based on a provided table of data.

## Google Generative AI and Gemini 1.5 Flash
## Google Generative AI

The google.generativeai library provides access to Google's advanced generative models, including the Gemini series. It enables integration of Google's powerful AI capabilities into various applications for text generation, content creation, and data understanding.

Library: `google.generativeai`

## Gemini 1.5 Flash
The gemini-1.5-flash model is a high-performance generative model optimized for rapid and accurate text generation and question answering. It is part of the Gemini family, known for its efficiency and effectiveness in handling various generative tasks.

Model Name: `gemini-1.5-flash`

## Features

- **Question Answering:** Allows users to query specific information from tabular data.
- **Integration with Google Gemini:** Utilizes the Gemini API to process and generate answers.
- **Dynamic Table Handling:** Supports large tables by converting them into a string format for the model.

## Usage

1. **Setup:**
   - Ensure you have your Google API key for Gemini. The API key is required to authenticate and use the Gemini model.

2. **Prepare the Data:**
   - Create a pandas DataFrame to represent the tabular data. The DataFrame should include columns relevant to your queries, such as "Product", "Price", "Brand", "Stock", and "Category".

3. **Formulate Your Question:**
   - Define a question related to the table data. For example, "What is the price and stock of the Printer?"

4. **Convert Table to String:**
   - Convert the DataFrame to a string format to be included in the prompt sent to the Gemini model.

5. **Generate the Answer:**
   - Use the Gemini API to generate an answer based on the table and the question. The model processes the prompt and returns a response.

6. **Print the Response:**
   - Output the generated answer to the console.

## Example

```python
import pandas as pd
import os
import google.generativeai as genai

# Step 1: Set up your Google API Key for Gemini
os.environ["GOOGLE_API_KEY"] = "Gemini api key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Step 2: Prepare a large table as a pandas DataFrame
data = {
    "Product": ["Laptop", "Smartphone", "Tablet", "Monitor", "Keyboard", "Mouse", "Printer", "Scanner", "Camera", "Speaker"],
    "Price": ["1000", "500", "300", "200", "50", "30", "150", "100", "400", "80"],
    "Brand": ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E", "Brand F", "Brand G", "Brand H", "Brand I", "Brand J"],
    "Stock": ["50", "200", "150", "80", "300", "500", "100", "75", "40", "60"],
    "Category": ["Electronics", "Electronics", "Electronics", "Accessories", "Accessories", "Accessories", "Office", "Office", "Photography", "Audio"]
}
table = pd.DataFrame.from_dict(data)

# Step 3: Prepare your question
question = "What is the price and stock of the Printer?"

# Step 4: Convert the DataFrame to a string format
table_str = table.to_string(index=False)

# Step 5: Generate content with Gemini
model = genai.GenerativeModel('gemini-1.5-flash')
prompt = f"Based on the following large table, answer the question: {question}\n\nTable:\n{table_str} without * "
response = model.generate_content(prompt)

# Step 6: Print the response
print(f"Question: {question}")
print(f"Answer: {response.text}")
```

## Requirements

- **Python 3.6+**
- **Pandas Library:** For creating and manipulating tabular data.
- **Google Gemini API:** To generate answers based on the table and question.

## Setup Instructions

1. **Install Dependencies:**
   Install the required Python libraries, including `pandas` and `google-generativeai`.

   ```bash
   pip install pandas google-generativeai



## Setup Instructions

1. **Install Dependencies:**
   Install the required Python libraries, including `pandas` and `google-generativeai`.

2. **Configure API Key:**
   Set up your Google API key for accessing the Gemini API.

3. **Run the Script:**
   Execute the script to perform tabular question answering.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Google Gemini API](https://cloud.google.com/generative-ai)
- [Pandas Library](https://pandas.pydata.org/)

# Tabular Question Answering on CSV data using Google GEMINI LLM



## Overview

This demonstrates how to use the Google Gemini API to perform question answering on data stored in a CSV file. The script leverages the `gemini-1.5-flash` model to answer questions based on the provided tabular data.

## Features

- **CSV Data Handling:** Loads and processes data from CSV files.
- **Question Answering:** Uses the Gemini API to generate answers based on the CSV data.
- **Integration with Google Gemini:** Utilizes the `gemini-1.5-flash` model for accurate and efficient responses.

## Prerequisites

- **Python 3.6+**
- **Pandas Library:** For loading and handling CSV data.
- **Google Gemini API:** To generate answers based on the CSV data.

## Installation

1. **Install Dependencies:**
   Install the required Python libraries using pip.

   ```bash
   pip install pandas google-generativeai
   ```

2. **Set Up API Key:**
   Replace `"Your gemini api key"` in the script with your actual Gemini API key.

## Usage

1. **Load CSV Data:**
   Modify the path to the CSV file in the script to point to your CSV file.

   ```python
   import pandas as pd

   # Load the CSV file into a DataFrame
   df = pd.read_csv(r'file.csv')

   # Display the first few rows of the DataFrame
   print(df.head())
   ```

2. **Configure Gemini API:**
   Set up the Gemini API key and load the `gemini-1.5-flash` model.

   ```python
   import google.generativeai as gemini

   # Set your Gemini API key
   gemini.configure(api_key="Your gemini api key")
   # Load the Gemini model
   model = gemini.GenerativeModel('gemini-1.5-flash')
   ```

3. **Ask a Question:**
   Define your question and call the function to get the answer.

   ```python
   def answer_question_with_gemini(question, df):
       # Convert the DataFrame to a string to use as context
       context = df.to_string(index=False)

       # Generate a response using the Gemini model
       response = model.generate_content(f'Answer the following question based on the data: {context}\n\nQuestion: {question}')
       
       # Return the response text
       return response.text

   # Example question
   question = "What is the age of pavan?"

   # Get the answer
   answer = answer_question_with_gemini(question, df)
   print(f"Answer: {answer}")
   ```

