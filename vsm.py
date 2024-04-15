import glob
import os
import string
from nltk.stem import PorterStemmer
import math 
import json
import tkinter as tk
import re 
import string
import time
from tkinter import messagebox
Dictionary = {} #Create a global dictionary
DocVectors = {} #Create a global dictionary for Document Vectors

def load_data():
    flag = False
    if os.path.exists('Dict.json'): #if Dictionary is already built
        flag = True
        with open('Dict.json', 'r') as f:
            Dictionary.update(json.load(f)) #read from Dict.json
    if os.path.exists('TFIDFVec.json'): #if Document Vectors are already built
        flag = True
        with open('TFIDFVec.json', 'r') as f:
            DocVectors.update(json.load(f))#read from TFIDFVec.json
    return flag



def FileRead(): 
    Folder = 'ResearchPapers'
    Pattern = '*.txt' 
    FList = glob.glob(os.path.join(Folder, Pattern)) #Finding all Files in the given Folder 
    for Path in FList: 
        with open(Path, 'r') as file: 
            FileContents = file.read() #Reading File text
            FileContents = FileContents.lower()
            File_name = Path.strip("ResearchPapers\\.txt")
            FileContents = PunctuationRemove(FileContents)# Removing Punctuations
            FileContents = FileContents.split() # Tokenizing string
            Stemmer = PorterStemmer()
            FileStem = []
            #Applying Stemming to all the tokens
            for words in list(FileContents):
                FileStem.append(Stemmer.stem(words))
            File_name = int(File_name)
            Dictionary = DictionaryBuilder(FileStem,File_name)
            Dictionary = sorted(Dictionary.items()) # Sorting the Dictionary by tokens
            Dictionary = dict(Dictionary)
    with open('Dict.json', 'w') as f:
        json.dump(Dictionary, f)  #saving Dicionary in Dict.json to skip building next time
    # Initializing all Document Vectors with 0 for every word
    for i in range(1,27):
        if (i == 4 or i==5 or i==6 or i==10 or i==19 or i==20): #Doc 4,5,6,10,19,20 dont exist in ResearchPapers Dataset
            continue
        DocVectors[i] = [0] * len(Dictionary)


    return Dictionary


def PunctuationRemove(File):
    File = File.replace('-', ' ') # Replacing hyphens with spaces
    File = File.translate(str.maketrans("", "", string.punctuation))
    File = re.sub(r'https?://(?:www\.)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', File)  # remove urls
    File = re.sub(r'\S+\.com\b', '', File)  # remove .com
    File = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', File)  # remove emails
    File = re.sub(r'[^\w\s]', '', File)  # remove other useless punctuation
    return File

def DictionaryBuilder(File,File_Name):
    Stop = open(r'Stopword-List.txt', 'r')
    StopContents = Stop.read()
    StopContents = StopContents.split()
    for words in File: # Building Dictionary
        if(words not in StopContents):
            if(words not in Dictionary): # First time a word is added to Dictionary
                Dictionary[words] = {}
                Dictionary[words][File_Name] = 1 # Setting Term Frequency for the document to 1
            else:
                if(File_Name not in Dictionary[words]):
                    Dictionary[words][File_Name] = 1 # Setting Term Frequency for the document to 1
                else:
                    Dictionary[words][File_Name] += 1 # Incrementing Term Frequency
    return Dictionary   

def BuildDocumentVectors():
    for Index, Key in enumerate(Dictionary): # Traversing through words in Dictionary
        for DocKeys in DocVectors.keys(): # Traversing through all Documents
            if(DocKeys in Dictionary[Key]):
                DocFreq = len(Dictionary[Key]) 
                InvertedDocFreq = round(math.log(len(DocVectors) / DocFreq, 10),2) # Calculating Inverted Document Frequency
                TfIdf = InvertedDocFreq * Dictionary[Key][DocKeys] 
                DocVectors[DocKeys][Index] = TfIdf
    
    with open('TFIDFVec.json', 'w') as f:
        json.dump(DocVectors, f) #saving TFIDFVec to skip building database next time

def QueryProcessor(Query):
    Query = Query.split()
    Query = QueryStemmer(Query)
    print(Query,end='\n')
    QueryVector = [0] * len(Dictionary) # Initializing Query Vector
    QueryDict = {}
    for words in Query: # Building Dictionary for Query
        if(words not in QueryDict): # First time a word is added to Dictionary
            QueryDict[words] = 1
        else:
            QueryDict[words] += 1
    for Index, Key in enumerate(Dictionary): # Traversing Dictionary
            if(Key in QueryDict):
                DocFreq = len(Dictionary[Key])
                InvertedDocFreq = math.log(len(DocVectors) / DocFreq, 10)
                TfIdf = InvertedDocFreq * QueryDict[Key] 
                QueryVector[Index] = TfIdf
    return QueryVector


def QueryStemmer(Query):
    StemQuery = []
    Stop = open(r'Stopword-List.txt', 'r')
    StopContents = Stop.read()
    StopContents = StopContents.split()
    Stemmer = PorterStemmer()
    Query = [Val for Val in Query if Val not in StopContents]
    for words in Query:
        StemQuery.append(Stemmer.stem(words))
    return StemQuery  

# Calculating Eucilidean Length for a Vector
def EucDist(Vector):
    Sum = 0
    for i in Vector:
        Sum += i ** 2
    return(math.sqrt(Sum))

def Solver(Query):
    ResultList = []
    QueryEucDist = EucDist(Query) # Calculating Eucilidean Length for the Query
    if(QueryEucDist == 0): # Return empty list if none of the words in the Query are in the Dictionary
        return ResultList
    for Doc in DocVectors.keys():
        Cosine = 0
        DotProduct = 0
        DocEucDist = EucDist(DocVectors[Doc]) # Calculating Eucilidean Length for a given Document
        if(DocEucDist == 0): # Return empty list if none of the words in the doc are in the Dictionary
            continue
        for i in range(0,len(Dictionary)):
            if(Query[i] == 0 or DocVectors[Doc][i] == 0): # Skip calculation if one of the TF-IDFs are 0
                continue
            else:
                DotProduct = DotProduct + (Query[i] * DocVectors[Doc][i])
        Cosine = DotProduct / (QueryEucDist * DocEucDist)
        if(Cosine > 0.01): # Threshold
            ResultList.append((Doc,Cosine))
    ResultList = sorted(ResultList, key=lambda x:-x[1]) # Sort results according to Cosine value
    return ResultList


def search_query():
    query = entry_query.get()
    query_vector = QueryProcessor(query)
    results = Solver(query_vector)

    if not results:
        result_label.config(text="No documents match the query.", fg="red")
    else:
        result_label.config(text="Search Results sorted by cosine similarity:", fg="black")
        result_text.delete('1.0', tk.END)
        for result in results:
            result_text.insert(tk.END, f"Document ID: {result[0]}, Cosine Similarity: {result[1]}\n")


# Check if the JSON files exist and load data
start2 = time.time() #start time if database build
sv= load_data()
print("Dictionary and DocVectors already exist?", sv)
if sv is False:
    start = time.time() #start time if database not built
    Dictionary = FileRead()
    BuildDocumentVectors()
    print("Time taken to build database: ", (time.time()-start), "seconds")
else:
    print("Time taken to load database: ",(time.time()- start2), "seconds")

root = tk.Tk()
root.title("Document Search Engine")
root.configure(bg="lavender")  # Change background color here
# Create label and entry for query input
label_query = tk.Label(root, text="Enter your query:", font=("Arial", 12))
label_query.grid(row=0, column=0, padx=10, pady=10, sticky="w")
entry_query = tk.Entry(root, width=50, font=("Arial", 12))
entry_query.grid(row=0, column=1, padx=10, pady=10)

# Create search button
button_search = tk.Button(root, text="Search", command=search_query, font=("Arial", 12))
button_search.grid(row=0, column=2, padx=10, pady=10)

# Create label and text widget for displaying search results
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.grid(row=1, column=0, columnspan=3, pady=(20, 10))

result_text = tk.Text(root, height=10, width=60, font=("Arial", 12))
result_text.grid(row=2, column=0, columnspan=3, padx=10)

# Run the GUI
root.mainloop()