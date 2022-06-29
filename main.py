from tkinter import *
import tkinter as tk
from tkinter import ttk
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import string
nltk.download("stopwords")

model = TFDistilBertForSequenceClassification.from_pretrained("https://drive.google.com/drive/folders/1exoJhG2Maj5sqjZ1802tjdTS46FRJtsr")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

app = Tk()
app.geometry("800x500")
app.title("Text Classification")

def Result():
    # tokenize input text
    input_ids = tokenizer(textEntry.get())
    # get tokens as a list of words
    tokens = tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0])
    # predict the label
    preds = model(input_ids)
    # show the prediction result
    sentiment = model.config.id2label[preds[0][0].numpy().argmax()]
  
    # Creates label and display Sentiment
    # Frame 1
    sentimentLabel = Label(sentimentFrame, text="The sentiment of above text is:")
    sentimentLabel.grid(column=0,row=0, padx= (0,10))
    sentimentButton = Button(sentimentFrame, text=sentiment)
    sentimentButton.grid(column=1,row = 0, padx= (10,0))
    sentimentFrame.grid(row=0,column=0)

    # filter out the only words that are not stopwords, punctuation, or numbers, cls token, and pad token
    stwrds = stopwords.words('english')
    # remove cls and sep tokens
    filtered_words = [word for word in tokens if word not in [
    '[CLS]', '[SEP]', stwrds, string.punctuation, string.digits, '.']]
    # plot frequency distribution of words with frequency greater than 1
    freq = nltk.FreqDist(filtered_words)
    # filter set where value is more than 1
    new_set = [(sub, val) for sub, val in freq.items() if val > 1]
    top = freq.most_common(4)
    # bold the words that are most common in the original text
    for sub, val in new_set:
        tokens = [word if word != sub else '**' + word + '**' for word in tokens]
    # final ouptut
    new_text = tokenizer.convert_tokens_to_string(tokens)
    # filter cls and sep tokens
    new_text = new_text.replace('[CLS]', '').replace('[SEP]', '')
    
    # Display the text with most frequent words bloded using new_text
    #Frame 2
    Label(resultFrame, text="The text with most frequent words highlighted:").grid(row=0,column=0)
    textResult = Label(resultFrame, text = new_text)
    textResult.grid(column=0,row=1)
    resultFrame.grid(row=1,column=0, rowspan=2)
    
    #Creates the "Word frequency" label
    #Frame 3
    freqLabel = Label(freqFrame, text="Word Frequency:")
    freqLabel.grid(row = 0,column = 0, padx = (0,90))
    freqFrame.grid(row = 0, column=1)
    
    # Display the word frequency table
    #Frame 4
    freqTable.columnconfigure(0,weight=1)
    freqTable.columnconfigure(1,weight=2)
    Label(freqTable,text="Word").grid(column=0,row=0,ipadx=40, columnspan=1)
    Label(freqTable,text="Frequency").grid(column=1,row=0, ipadx=40, columnspan=1)
    Label(freqTable,text=top[0][0]).grid(column=0,row=1, ipadx=40, columnspan=1)
    Label(freqTable,text=top[0][1]).grid(column=1,row=1, ipadx=40, columnspan=1)
    Label(freqTable,text=top[1][0]).grid(column=0,row=2, ipadx=40, columnspan=1)
    Label(freqTable,text=top[1][1]).grid(column=1,row=2, ipadx=40, columnspan=1)
    Label(freqTable,text=top[2][0]).grid(column=0,row=3, ipadx=40, columnspan=1)
    Label(freqTable,text=top[2][1]).grid(column=1,row=3, ipadx=40, columnspan=1)
    Label(freqTable,text=top[3][0]).grid(column=0,row=4, ipadx=40, columnspan=1)
    Label(freqTable,text=top[3][1]).grid(column=1,row=4, ipadx=40, columnspan=1)
    freqTable.grid(row=1,column=1)

#upper body

textEntry = Entry(app)
textEntry.place(x = 175, y = 40, width = 500, height = 150)

    #Show result button take the text entered
    #and pass it to tokenier in result function
submitButton = Button(app, text="Show Results", command=Result)
submitButton.place(x = 375, y = 210)


#lower Body

    #Main frame
frame = ttk.Frame(app, width= 500, height=150)
    #Frame 1: show the sentiment positive or negative
sentimentFrame = ttk.Frame(frame)
    #Frame 2: display the bolded text
resultFrame = ttk.Frame(frame)
    #Frame 3: "Word frequency" label
freqFrame = ttk.Frame(frame)

    #Frame 4: word frequency table
freqTable = ttk.Frame(frame)



frame.place(x = 175, y = 300)
app.mainloop()