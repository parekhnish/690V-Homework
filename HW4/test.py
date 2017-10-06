import numpy as np
import pandas as pd

all_data = pd.read_csv("Datasets/nutrition_raw_anonymized_data.csv")
num_rows = len(list(all_data.index))
num_cols = len(list(all_data.columns))

all_columns = list(all_data.columns)


produce_names = ['BROCCOLI','CARROTS','CORN','GREENBEANS','COOKEDGREENS','CABBAGE','GREENSALAD','RAWTOMATOES','SALADDRESSINGS','AVOCADO','SWEETPOTATOES','FRIES','POTATOES','OTHERVEGGIES','MELONS','BERRIES','BANANAS','APPLES','ORANGES','PEACHES','OTHERFRESHFRUIT','DRIEDFRUIT','CANNEDFRUIT','REFRIEDBEANS','BEANS','TOFU','MEATSUBSTITUTES','LENTILSOUP','VEGETABLESOUP','OTHERSOUP']
meat_names = ['HAMBURGER','HOTDOG','BACONSAUSAGE','LUNCHMEAT','MEATBALLS','STEAK','TACO','RIBS','PORKCHOPS','BEEFPORKDISH','LIVER','VARIETYMEAT','VEALLAMBGAME','FRIEDORBREADEDCHICKEN','ROASTCHICKEN','OTHERCHICKENDISH','OYSTERS','SHELLFISH','TUNA','SALMON','FRIEDORBREADEDFISH','OTHERFISH']


for name in produce_names:
    all_data[name+"_TOTAL"] = all_data.loc[:,name+"FREQ"] * all_data.loc[:,name+"QUAN"]

for name in meat_names:
    all_data[name+"_TOTAL"] = all_data.loc[:,name+"FREQ"] * all_data.loc[:,name+"QUAN"]


print(all_data.loc[:,"STEAK_TOTAL"])