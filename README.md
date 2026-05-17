# Part 3: NLP and Sequence Modeling Mini Project

## Task 1: Dataset Understanding

The text dataset was used for sentiment classification. It contained 1,500 records and 6 columns.

The target variable was `sentiment_label`, with three classes: `neutral`, `negative`, and `positive`.

No missing values were found in the dataset. The class distribution was fairly balanced, with 524 neutral, 497 negative, and 479 positive records.

The average text length was about 72.76 characters, and the average word count was about 12.72 words.


## Task 2: Text Preprocessing

The text data was cleaned before modeling. All messages were converted to lowercase, and unnecessary symbols or special characters were removed.

The cleaned text was tokenized so that words could be converted into numerical form. The tokenized sequences were padded and truncated to a fixed length of 30.

After preprocessing, the padded sequence data had a shape of 1,500 × 30.


## Task 3: Text Vectorization

The cleaned text was converted into numerical format using TF-IDF and tokenizer-based sequences.

The TF-IDF representation created a feature matrix with a shape of 1,500 × 180. Tokenizer-based sequences were also created with a fixed shape of 1,500 × 30.

Text was converted into numerical form because machine learning models cannot directly understand raw text. This step allowed the model to learn patterns from the words in each customer message.


## Task 4: Baseline Model

A Logistic Regression model was used as the baseline model with TF-IDF features.

The dataset was split into training and testing sets using an 80:20 split. The baseline model achieved 100% testing accuracy.

The confusion matrix showed that all 300 test records were correctly classified. The model correctly predicted 99 negative, 105 neutral, and 96 positive messages.


## Task 5: Sequence Model or Conceptual Architecture

A simple LSTM-based sequence model was built using the padded token sequences.

The input sequence had a fixed length of 30 tokens. An embedding layer was used to convert each token into a dense vector representation. An LSTM layer was used to learn word order and sequence patterns from the messages.

A dropout layer and dense layer were added before the final output layer. The output layer used softmax activation because there were three sentiment classes.

Categorical crossentropy was used as the loss function, and accuracy was used as the evaluation metric. The LSTM model achieved a testing accuracy of 100%.


## Task 6: Attention and Transformer Reflection

RNNs can struggle with long-term dependencies because earlier information may become less prominent as the sequence becomes longer.

LSTMs help with this problem by using gates to control what information should be remembered or forgotten. This makes them better at handling longer text sequences than simple RNNs.

Attention helps a model focus on the most important words in a sequence instead of treating all words equally. This is useful in sequence-to-sequence tasks such as translation and summarization.

Transformers are important in modern NLP because they use attention to process text more effectively and can handle long sequences in parallel. They are also the foundation for many generative AI models.
