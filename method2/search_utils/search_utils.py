"""<u>Observations</u> :
* The training curves indicates that the model needs to be learned more time but also that the model may nt be suitted to the dataset or the inverse. In the same way, when a model learns to enough, this can be a cause of underfitting.

# 6 - Greedy and Beam Search

As the model generates a 2101 long vector with a probability distribution across all the words in the vocabulary we greedily pick the word with the highest probability to get the next word prediction. This method is called Greedy Search.
"""

import numpy as np
from keras.preprocessing.sequence import pad_sequences

class SearchUtils():
    def greedySearch(photo,model, max_length, wordtoix,ixtoword):
        in_text = 'startseq'
        for i in range(max_length):
            sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([photo, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break

        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final


    """Beam Search is where we take top k predictions, feed them again in the model and then sort them using the probabilities returned by the model. So, the list will always contain the top k predictions and we take the one with the highest probability and go through it till we encounter ‘endseq’ or reach the maximum caption length."""


    def beam_search_predictions(image, model, max_length, wordtoix, ixtoword, beam_index=3):
        start = [wordtoix["startseq"]]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < max_length:
            temp = []
            for s in start_word:
                par_caps = pad_sequences([s[0]], maxlen=max_length, padding='post')
                preds = model.predict([image, par_caps], verbose=0)
                word_preds = np.argsort(preds[0])[-beam_index:]
                # Getting the top <beam_index>(n) predictions and creating a
                # new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]

        start_word = start_word[-1][0]
        intermediate_caption = [ixtoword[i] for i in start_word]
        final_caption = []

        for i in intermediate_caption:
            if i != 'endseq':
                final_caption.append(i)
            else:
                break

        final_caption = ' '.join(final_caption[1:])
        return final_caption
