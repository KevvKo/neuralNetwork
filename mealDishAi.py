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
        '''TO DO's:
            define object properties (words, intents, classes,...)
            - function createAlphabet
            - function prediction
            - function decodeJsonFile
            - trainModel
        '''
        #self._intents = []
        #self.words = pickle.load(open('src/words.pkl', 'rb'))

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

    #decodes the json file(recips.json) and creating wordbags for the AI
    def decodeJSonFile(self):
        pass

    #creating an unique alphabet with words, which are containted in the dataset
    def creatingAlphabet(self):
        words = []
        classes = []
        ignoreWords = ['?', '!']

        with open('src/recipes.json', 'r') as file:
            dataFile = json.load(file)

        for intent in dataFile['intents']:
            for recipe in intent:
                w = nltk.word_tokenize(recipe)
                words.extend(w)

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignoreWords]

        sorted(list(set(words)))
        pickle.dump(words,open('src/words.pkl','wb'))

    #make a prediction and return the computed match
    def prediction(self):
        pass
    
    #train the model and save the result
    def trainModel(self):
        pass

###################################################################################################################

if __name__ == "__main__":

    bot = MealDishAi()
    #bot.createRecipesFile()
    bot.loadJson('src/recipes.json')
    bot.creatingAlphabet()