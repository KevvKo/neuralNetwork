from network import Network
import json
import sys

class MealDishAi(Network):

    def __init__(self):
        pass

    #decodes the available json with receipes and ingredients
    def recipesDecoder(self):
        with open ('src/chefkoch.json', 'r') as file:
            data = json.load(file)
        
        js = []
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

        js.append({'intents': intents})
        
        # create the decoded json with recipes
        with open('src/recipes.json', 'w') as file:
            json.dump(js, file, indent=4)

#######################################################################################

if __name__ == "__main__":

    bot = MealDishAi()
    
    bot.loadJson('src/recipes.json')