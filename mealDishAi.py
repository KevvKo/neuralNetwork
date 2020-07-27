from network import Network
import json
import sys
import nltk
import numpy as np 
import pickle

from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

class MealDishAi(Network):

    def __init__(self):

        self._intents = json.loads(open('src/recipes.json').read())
        self._words = pickle.load(open('src/words.pkl', 'rb'))
        self._classes = pickle.load(open('src/classes.pkl', 'rb'))


    #creating a reusable json-file with recipes and further properties
    def createRecipesFile(self):
        with open ('src/chefkoch.json', 'r') as file:
            data = json.load(file)
        
        intents = []
        
        # loop through the rawdata and put the ingredionts and instructions for every meal and 
        # create a single jsonline to append to the new decoded dataset
        for i in range(len(data)):
            
            # intercept the necessary informations
            jsline = {}
            
            mealName = data[i]['Name']
            preparation =  data[i]['Instructions']
            ingredients = data[i]['Ingredients']

            # the line with the informations for any meal
            jsline = {
                'preparation': preparation,
                'ingredients': ingredients
            }

            # the created intents-dataset
            intents.append(
                {mealName: jsline}
            )
        
        # create the decoded json with recipes
        with open('src/recipes.json', 'w') as file:
            json.dump({'intents': intents}, file, indent=4)


    #creating an unique alphabet with words, which are containted in the dataset and
    #further a sorted list of all classes
    def creatingAlphabet(self):
        words = []
        classes = []
        ignoreWords = ['?', '!']

        with open('src/recipes.json', 'r') as file:
            dataFile = json.load(file)

        #loop through the dataset and creating all necessary lists
        for intent in dataFile['intents']:
            for recipe in intent:
                w = nltk.word_tokenize(recipe)
                words.extend(w)

                if recipe not in classes: classes.append(recipe)

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignoreWords]

        #save the sorted list of unique words in a binary file (pickle) to archive them
        sorted(list(set(words)))
        pickle.dump(words,open('src/words.pkl','wb'))

        #save the classes as a sorted list in a pkl-file
        sorted(list(set(classes)))
        pickle.dump(classes, open('src/classes.pkl', 'wb'))
        
    #creating bag of words
    wordbag(self):
        pass
    
    #train the model and save the result
    def trainModel(self):
        pass
    
    #make a prediction and return the computed match
    def prediction(self):
        pass

    def getResponse(self):
        pass

    def cleanUpSentence(self):
        pass
    
    def run(self):
        pass

###################################################################################################################

if __name__ == "__main__":

    bot = MealDishAi()
    #bot.createRecipesFile()
    bot.loadJson('src/recipes.json')
    bot.creatingAlphabet()