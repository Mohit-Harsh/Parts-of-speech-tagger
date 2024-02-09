# Parts-of-speech-tagger

https://github.com/Mohit-Harsh/Parts-of-speech-tagger/assets/111693866/71971165-9b9c-42bf-9d71-e53bd8d4f78d

Parts of Speech Tagger is a Streamlit Application that takes a Sentence as input and generates a POS(parts of speech) Tag for each word/token in the sentence. The model uses LSTM architechture to generate the pos tags.

# Libraries Required

Some important libraries required for building this project are:

  * <img src="https://github.com/Mohit-Harsh/News-Headline-Tagger/blob/main/assets/tensorflow.png" height="30" align="center"/> Tensorflow
  * <img src="https://github.com/Mohit-Harsh/News-Headline-Tagger/blob/main/assets/keras.png" height="30" align="center"/> Keras
  * <img src="https://github.com/Mohit-Harsh/News-Headline-Tagger/blob/main/assets/icons8-python-96.png" height="30" align="center"/> Tkinter
  * <img src="https://github.com/Mohit-Harsh/News-Headline-Tagger/blob/main/assets/pandas.png" height="30" align="center"/> Pandas
  * <img src="https://github.com/Mohit-Harsh/News-Headline-Tagger/blob/main/assets/numpy.png" height="30" align="center"/> Numpy
  * <img src="https://github.com/Mohit-Harsh/News-Headline-Tagger/blob/main/assets/icons8-python-96.png" height="30" align="center"/> BeautifulSoup
    
For further details about the requirements refer the requirements.txt file.

# How to Build

### 1. Preprocess

* Download the data - corpus from nltk.
* Initialize and fit the Tokenizers on the downloaded corpus:
  
    * Sentence_Tokenizer - to tokenize input texts
    * Tag_Tokenizer - to tokenize output tags
      
* Convert texts to sequences.
* One hot encode output tags.
* Split the data into Train, Test and Validation sets.
* Pad the Sequences to max length.

* Refer [POS Tagger.ipynb](https://github.com/Mohit-Harsh/Parts-of-speech-tagger/blob/391d7424e82f45228c303aa629a2aa81b63278ec/POS%20Tagger.ipynb) file for python code.

### 2. Build

* Build a sequential model with the following components:

    * EmbeddingLayer - 128 dim, input length - max sequence length
    * LSTM layer - input dim - 128, return sequences should be True
    * Dense layer - activation - Softmax, units = no. of POS tags = 13
      
* Compile the model.
* Train the model.
* Save the model and the tokenizer.

* Refer [POS Tagger.ipynb](https://github.com/Mohit-Harsh/Parts-of-speech-tagger/blob/391d7424e82f45228c303aa629a2aa81b63278ec/POS%20Tagger.ipynb) file for python code.

### 3. Deploy

* Load the saved Tokenizers and LSTM model.
* Take a list of POS tokens in the same sequence as the tag_tokenizer.
* Tokenize the input sentence using the Sentence_Tokenizer and then convert it into sequences.
* Give the sequences as input to the LSTM model predict function.
* The output consists of output of every time step, so take the output of the last time step by negative indexing : result[-1].
* Use np.argmax() to get the tag index with highest probability.
* Use list indexing to get the corresponding tag name.
* You can also use streamlit to deploy the model as a Web Application.

* Refer []
* Refer [streamlit-app.py](https://github.com/Mohit-Harsh/Parts-of-speech-tagger/blob/main/streamlit-app.py) file for streamlit application python code.

