from network import Network
import json
import sys
import nltk
import numpy as np 
import pickle

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

class MealDishAi(Network):

    def __init__(self):
        pass

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

    def buildAlphabet(self):
        words = []
#######################################################################################

if __name__ == "__main__":

    bot = MealDishAi()
    bot.createRecipesFile()
    bot.loadJson('src/recipes.json')