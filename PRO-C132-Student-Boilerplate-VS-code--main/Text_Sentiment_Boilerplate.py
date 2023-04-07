import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ["I am happy to meet my friends. We are planning to go a party.", 
            "I had a bad day at school. i got hurt while playing football"]

# Tokenization
tokenizer = Tokenizer(num_words = 10000, oov_token = '<OOV>')
tokenizer.fit_on_texts(sentence)

# Create a word_index dictionary
word_index = tokenizer.word.index
sequence =  tokenizer.texts(sentence)

# Padding the sequence
padded = pad_sequences(sequence,maxlen = 100,padding='post', truncating='post')
# Define the model using .h5 file
model = tenserflow.keras.models.load_model('text_emotion.h5')
# Test the model
result = model.predict(padded)
# Print the result
predict_class = mp.argmax(result, axis = 1)

