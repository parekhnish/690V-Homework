import numpy as np
import pandas as pd

from bokeh.layouts import column, row, widgetbox
from bokeh.palettes import Category20
from bokeh.plotting import figure, show, curdoc
from bokeh.models import ColumnDataSource, Select, MultiSelect

from helper_functions import *




all_data = pd.read_csv("Datasets/nutrition_raw_anonymized_data.csv")
num_rows = len(list(all_data.index))
num_cols = len(list(all_data.columns))

all_columns = list(all_data.columns)



g = all_data.columns.to_series().groupby(all_data.dtypes).groups
temp_dict = {k.name: v for k,v in g.items()}
object_columns = list(temp_dict['int64'])
num_unique_elems = [len(list(all_data[col].unique())) for col in object_columns]
binary_cols = [i for i,j in enumerate(num_unique_elems) if j==2]


color_dict = {i: Category20[20][i*2] for i in range(9)}



all_data['bool_disease'] = pd.Series(makeBoolColumn(all_data,['cancer','diabetes','heart_disease'],'Yes'))
all_data['type_smoking'] = pd.Series(makeTypeColumn(all_data,['smoke_rarely','smoke_often'],'Yes',False))
all_data['bool_smoking'] = pd.Series(makeBoolColumn(all_data,['smoke_rarely','smoke_often'],'Yes'))
all_data['type_hand'] =    pd.Series(makeTypeColumn(all_data,['left_hand','right_hand'],'Yes',True))
all_data['type_pisa'] =    pd.Series(makeTypeColumn(all_data,['readingMath','mathReading'],'Yes',False))
all_data['type_cable'] =   pd.Series(makeTypeColumn(all_data,['unfavCable','neutralCable','favCable'],'Yes',True))
all_data['type_crash'] =   pd.Series(makeTypeColumn(all_data,['noCrash','uhCrash','yesCrash'],'Yes',True))
all_data['type_pet'] =     pd.Series(makeTypeColumn(all_data,['cat','dog'],'Yes',False))
all_data['bool_pet'] =     pd.Series(makeBoolColumn(all_data,['cat','dog'],'Yes'))
all_data['bool_belly'] =   pd.Series(makeBoolColumn(all_data,['belly'],'Outie'))
all_data['bool_hist_smoked'] = pd.Series(makeBoolColumn(all_data,['ever_smoked'],'Yes'))
all_data['bool_rash'] =    pd.Series(makeBoolColumn(all_data,['rash'],'Yes'))
all_data['type_race'] =    pd.Series(makeTypeColumn(all_data,['LATINO','WHITE','BLACK','ASIAN','NATIVEAMER','HAWAIIAN'],1,False))
all_data['bool_cancer'] =  pd.Series(makeBoolColumn(all_data,['cancer'],'Yes'))
all_data['bool_diabetes'] =  pd.Series(makeBoolColumn(all_data,['diabetes'],'Yes'))
all_data['bool_heart_disease'] =  pd.Series(makeBoolColumn(all_data,['heart_disease'],'Yes'))


veg_names = ['BROCCOLI','CARROTS','CORN','GREENBEANS','COOKEDGREENS','CABBAGE','GREENSALAD','RAWTOMATOES','SALADDRESSINGS','AVOCADO','SWEETPOTATOES','FRIES','POTATOES','OTHERVEGGIES','MELONS','BERRIES','BANANAS','APPLES','ORANGES','PEACHES','OTHERFRESHFRUIT','DRIEDFRUIT','CANNEDFRUIT','REFRIEDBEANS','BEANS','TOFU','MEATSUBSTITUTES','LENTILSOUP','VEGETABLESOUP','OTHERSOUP']
meat_names = ['HAMBURGER','HOTDOG','BACONSAUSAGE','LUNCHMEAT','MEATBALLS','STEAK','TACO','RIBS','PORKCHOPS','BEEFPORKDISH','LIVER','VARIETYMEAT','VEALLAMBGAME','FRIEDORBREADEDCHICKEN','ROASTCHICKEN','OTHERCHICKENDISH','OYSTERS','SHELLFISH','TUNA','SALMON','FRIEDORBREADEDFISH','OTHERFISH']


for name in veg_names:
    all_data[name+"_TOTAL"] = all_data.loc[:,name+"FREQ"] * all_data.loc[:,name+"QUAN"]

for name in meat_names:
    all_data[name+"_TOTAL"] = all_data.loc[:,name+"FREQ"] * all_data.loc[:,name+"QUAN"]



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


cat_label_dict = {}

cat_label_dict['bool_disease'] = {0: 'No Disease', 1: 'Diseased'}
cat_label_dict['type_smoking'] = {0: 'No Smoking', 1: 'Smoke Rarely', 2: 'Smoke Often'}
cat_label_dict['bool_smoking'] = {0: 'No Smoking', 1: 'Smoking'}
cat_label_dict['type_hand'] = {0: 'Left-Handed', 1: 'Right-Handed'}
cat_label_dict['type_pisa'] = {0: 'Science', 1: 'Reading', 2: 'Math'}
cat_label_dict['type_cable'] = {0: 'Not Favourite-Cable', 1: 'Neutral-Cable', 2: 'Favourite-Cable'}
cat_label_dict['type_crash'] = {0: 'No Crash', 1: 'Uh Crash', 2: 'Yes Crash'}
cat_label_dict['type_pet'] = {0: 'No Pet', 1: 'Cat', 2: 'Dog'}
cat_label_dict['bool_pet'] = {0: 'No Pet', 1: 'Has Pet'}
cat_label_dict['bool_belly'] = {0: 'Innie', 1: 'Outie'}
cat_label_dict['bool_hist_smoked'] = {0: 'Never Smoked', 1: 'Has Smoked'}
cat_label_dict['bool_rash'] = {0: 'No Rash', 1: 'Has Rash'}
cat_label_dict['type_race'] = {0: 'Not Specified', 1: 'Latino', 2: 'White', 3: 'Black', 4: 'Asian', 5: 'Native American', 6: 'Hawaiian'}
cat_label_dict['bool_cancer'] =  {0: 'No Cancer' , 1: 'Cancer'}
cat_label_dict['bool_diabetes'] =  {0: 'No Diabetes' , 1: 'Diabetes'}
cat_label_dict['bool_heart_disease'] = {0: 'No Heart Disease' , 1: 'Heart Disease'}

inv_cat_label_dict = {k: {n: i for (i,n) in cat_label_dict[k].items()} for k,d in cat_label_dict.items()}

cat_column_names_dict = {
                            '(Boolean) Disease': 'bool_disease',
                            '(Boolean) Cancer': 'bool_cancer',
                            '(Boolean) Diabetes': 'bool_diabetes',
                            '(Boolean) Heart Disease': 'bool_heart_disease',
                            '(Boolean) Currently Smoking': 'bool_smoking',
                            '(Boolean) Historical Smoking': 'bool_hist_smoked',
                            '(Boolean) Pets': 'bool_pet',
                            '(Boolean) Belly': 'bool_belly',
                            '(Boolean) Rash': 'bool_rash',
                            '(Type of) Smoking': 'type_smoking',
                            '(Type of) Handedness': 'type_hand',
                            '(Type of) Pisa Best Score': 'type_pisa',
                            '(Type of) Cable Favoritism': 'type_cable',
                            '(Type of) Crash': 'type_crash',
                            '(Type of) Pet': 'type_pet',
                            '(Type of) Race': 'type_race'
                            
}



cat_cat_params_dict = {
                    'x_name': list(cat_column_names_dict.values())[0],
                    'y_name': list(cat_column_names_dict.values())[1],
                    'target_name': list(cat_column_names_dict.values())[2],
                    'radius_scale': 1
}

cat_cat_data_dict = makeDataDictForCatCat(cat_cat_params_dict , list(cat_label_dict[cat_cat_params_dict['target_name']].values()) , all_data , color_dict, cat_label_dict)
cat_cat_CDS = ColumnDataSource(data=cat_cat_data_dict)

cat_cat_tree = makeTreeForCatCat(cat_cat_params_dict , all_data , cat_label_dict)
cat_cat_background_dict = makeDataDictForBackgroundCatCat(cat_cat_params_dict , cat_cat_tree , all_data , color_dict, cat_label_dict)
cat_cat_background_CDS = ColumnDataSource(data=cat_cat_background_dict)

cat_cat_prec_rec_data_dict = makeDataDictForCatCatPrecRec(cat_cat_params_dict , cat_cat_tree , 0 , all_data , color_dict , cat_label_dict)
cat_cat_prec_rec_CDS = ColumnDataSource(data=cat_cat_prec_rec_data_dict)




cat_cat_x_select = Select(options=list(cat_column_names_dict.keys()) , value=list(cat_column_names_dict.keys())[0] , title="X Data")
cat_cat_y_select = Select(options=list(cat_column_names_dict.keys()) , value=list(cat_column_names_dict.keys())[1] , title="Y Data")
cat_cat_target_select = Select(options=list(cat_column_names_dict.keys()) , value=list(cat_column_names_dict.keys())[2] , title="Target Variable")

cat_cat_mult_select = MultiSelect(options=list(cat_label_dict[cat_cat_params_dict['target_name']].values()),value=list(cat_label_dict[cat_cat_params_dict['target_name']].values()),title="Selected Targets")

cat_cat_prec_rec_val_select = Select(options=list(inv_cat_label_dict[cat_cat_params_dict['target_name']].keys()) , value=list(inv_cat_label_dict[cat_cat_params_dict['target_name']].keys())[0] , title="Prec-Rec Class")

cat_cat_options_box = widgetbox(cat_cat_x_select , cat_cat_y_select , cat_cat_target_select , cat_cat_mult_select)
cat_cat_prec_rec_options_box = widgetbox(cat_cat_prec_rec_val_select)


def catCatChangeData():

    global cat_cat_tree
    # print("ENTER -------- ChangeData ---------")
    cat_cat_CDS.data = makeDataDictForCatCat(cat_cat_params_dict , cat_cat_mult_select.value, all_data , color_dict, cat_label_dict)
    cat_cat_tree = makeTreeForCatCat(cat_cat_params_dict , all_data , cat_label_dict)
    cat_cat_background_CDS.data = makeDataDictForBackgroundCatCat(cat_cat_params_dict , cat_cat_tree , all_data , color_dict, cat_label_dict)
    # print("EXIT -------- ChangeData ---------\n-----------------")

    
def catCatPrecRecChangeData():

    global cat_cat_tree
    # print("ENTER -------- PrecRecChangeData ---------")
    cat_cat_prec_rec_CDS.data = makeDataDictForCatCatPrecRec(cat_cat_params_dict , cat_cat_tree , inv_cat_label_dict[cat_cat_params_dict['target_name']][cat_cat_prec_rec_val_select.value] , all_data , color_dict , cat_label_dict)
    # print("EXIT -------- PrecRecChangeData ---------\n-----------------")

def catCatAxesCallback(attrname,old,new):
    # print("ENTER -------- AxesCallback ---------")
    cat_cat_params_dict['x_name'] = cat_column_names_dict[cat_cat_x_select.value]
    cat_cat_params_dict['y_name'] = cat_column_names_dict[cat_cat_y_select.value]

    catCatChangeData()
    catCatPrecRecChangeData()
    
    cat_cat_figure.x_range.factors = list(cat_label_dict[cat_cat_params_dict['x_name']].values())
    cat_cat_figure.y_range.factors = list(cat_label_dict[cat_cat_params_dict['y_name']].values())
    # print("EXIT -------- AxesCallback ---------\n-----------------")

        
def catCatTargetSelectCallback(attrname,old,new):
    # print("ENTER -------- TargetCallback ---------")
    cat_cat_params_dict['target_name'] = cat_column_names_dict[cat_cat_target_select.value]
    
    # print("\t\tadded_target_name")
    cat_cat_mult_select.options = list(cat_label_dict[cat_column_names_dict[cat_cat_target_select.value]].values())
    cat_cat_mult_select.value = list(cat_label_dict[cat_column_names_dict[cat_cat_target_select.value]].values())

    # print("\t\tchanged_options")

    cat_cat_prec_rec_val_select.options = list(inv_cat_label_dict[cat_cat_params_dict['target_name']].keys())
    cat_cat_prec_rec_val_select.value = list(inv_cat_label_dict[cat_cat_params_dict['target_name']].keys())[0]
    # print("\t\tchange_prec_rec_options")
    
    catCatChangeData()
    catCatPrecRecChangeData()
    # print("EXIT -------- TargetCallback ---------\n-----------------")
        

def catCatMultSelectCallback(attrname,old,new):
    # print("ENTER -------- MultSelectCallback ---------")
    catCatChangeData()
    # print("EXIT -------- MultSelectCallback ---------\n-----------------")
        
        

def catCatPrecRecValCallback(attrname,old,new):
    # print("ENTER -------- PrecRecCallback ---------")
    catCatPrecRecChangeData()
    # print("EXIT -------- PrecRecCallback ---------\n-----------------")
        
    
cat_cat_x_select.on_change("value",catCatAxesCallback)
cat_cat_y_select.on_change("value",catCatAxesCallback)
cat_cat_target_select.on_change("value",catCatTargetSelectCallback)
cat_cat_mult_select.on_change("value",catCatMultSelectCallback)
cat_cat_prec_rec_val_select.on_change("value",catCatPrecRecValCallback)





cat_cat_figure = figure(title="Category-Category Scatter Plot",
                        plot_width=500,plot_height=450,
                        x_range=list(cat_label_dict[cat_cat_params_dict['x_name']].values()),
                        y_range=list(cat_label_dict[cat_cat_params_dict['y_name']].values()))

cat_cat_figure.circle(x='x_data' , y='y_data' ,
                      fill_color='color_data' , line_color='color_data' , fill_alpha=0.2 , line_alpha=0.2 ,
                      radius=0.01 , 
                      source=cat_cat_background_CDS)

cat_cat_figure.circle(x='x_data' , y='y_data' ,
                      fill_color='color_data' , line_color='color_data' , fill_alpha=0.8 ,
                      radius='radius_data' , legend='legend_data' ,
                      source=cat_cat_CDS)


cat_cat_prec_rec_figure = figure(title="Category-Category Prec-Rec Plot",
                                 plot_width=500,plot_height=450,
                                 x_range=['T-Act','F-Act'],
                                 y_range=['F-Pred','T-Pred'])

cat_cat_prec_rec_figure.quad(top='t_data' , bottom='b_data' , left='l_data' , right='r_data', 
                               fill_color='color_data' , line_color='color_data' , fill_alpha=0.8 ,
                               source=cat_cat_prec_rec_CDS)



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



veg_meat_params_dict = {
                        'x_name': veg_names[0],
                        'y_name': meat_names[1],
                        'target_name': list(cat_column_names_dict.values())[0],
                        'radius_scale': 10
}


veg_meat_data_dict = makeDataDictForVegMeat(veg_meat_params_dict , list(cat_label_dict[veg_meat_params_dict['target_name']].values()) , all_data , color_dict, cat_label_dict)
veg_meat_CDS = ColumnDataSource(data=veg_meat_data_dict)

veg_meat_tree = makeTreeForVegMeat(veg_meat_params_dict , all_data)
veg_meat_background_dict = makeDataDictForBackgroundVegMeat(veg_meat_params_dict , veg_meat_tree , all_data , color_dict)
veg_meat_background_CDS = ColumnDataSource(data=veg_meat_background_dict)

veg_meat_prec_rec_data_dict = makeDataDictForVegMeatPrecRec(veg_meat_params_dict , veg_meat_tree , 0 , all_data , color_dict)
veg_meat_prec_rec_CDS = ColumnDataSource(data=veg_meat_prec_rec_data_dict)




veg_meat_x_select = Select(options=veg_names , value=veg_names[0] , title="X Data")
veg_meat_y_select = Select(options=meat_names , value=meat_names[0] , title="Y Data")
veg_meat_target_select = Select(options=list(cat_column_names_dict.keys()) , value=list(cat_column_names_dict.keys())[0] , title="Target Variable")

veg_meat_mult_select = MultiSelect(options=list(cat_label_dict[veg_meat_params_dict['target_name']].values()),value=list(cat_label_dict[veg_meat_params_dict['target_name']].values()),title="Selected Targets")

veg_meat_prec_rec_val_select = Select(options=list(inv_cat_label_dict[veg_meat_params_dict['target_name']].keys()) , value=list(inv_cat_label_dict[veg_meat_params_dict['target_name']].keys())[0] , title="Prec-Rec Class")

veg_meat_options_box = widgetbox(veg_meat_x_select , veg_meat_y_select , veg_meat_target_select , veg_meat_mult_select)
veg_meat_prec_rec_options_box = widgetbox(veg_meat_prec_rec_val_select)




def vegMeatChangeData():

    global veg_meat_tree
    # print("ENTER -------- ChangeData ---------")
    veg_meat_CDS.data = makeDataDictForVegMeat(veg_meat_params_dict , veg_meat_mult_select.value, all_data , color_dict, cat_label_dict)
    veg_meat_tree = makeTreeForVegMeat(veg_meat_params_dict , all_data)
    veg_meat_background_CDS.data = makeDataDictForBackgroundVegMeat(veg_meat_params_dict , veg_meat_tree , all_data , color_dict)
    # print("EXIT -------- ChangeData ---------\n-----------------")

    
def vegMeatPrecRecChangeData():

    global veg_meat_tree
    # print("ENTER -------- PrecRecChangeData ---------")
    veg_meat_prec_rec_CDS.data = makeDataDictForVegMeatPrecRec(veg_meat_params_dict , veg_meat_tree , inv_cat_label_dict[veg_meat_params_dict['target_name']][veg_meat_prec_rec_val_select.value] , all_data , color_dict)
    # print("EXIT -------- PrecRecChangeData ---------\n-----------------")

def vegMeatAxesCallback(attrname,old,new):
    # print("ENTER -------- AxesCallback ---------")
    veg_meat_params_dict['x_name'] = veg_meat_x_select.value
    veg_meat_params_dict['y_name'] = veg_meat_y_select.value

    vegMeatChangeData()
    vegMeatPrecRecChangeData()

    x_range = np.max(all_data.loc[:,veg_meat_params_dict['x_name']+"_TOTAL"]) - np.min(all_data.loc[:,veg_meat_params_dict['x_name']+"_TOTAL"])
    y_range = np.max(all_data.loc[:,veg_meat_params_dict['y_name']+"_TOTAL"]) - np.min(all_data.loc[:,veg_meat_params_dict['y_name']+"_TOTAL"])

    veg_meat_figure.x_range.start = np.min(all_data.loc[:,veg_meat_params_dict['x_name']+"_TOTAL"]) - (x_range*0.1)
    veg_meat_figure.x_range.end = np.max(all_data.loc[:,veg_meat_params_dict['x_name']+"_TOTAL"]) + (x_range*0.1)
    veg_meat_figure.y_range.start = np.min(all_data.loc[:,veg_meat_params_dict['y_name']+"_TOTAL"]) - (y_range*0.1)
    veg_meat_figure.y_range.end = np.max(all_data.loc[:,veg_meat_params_dict['y_name']+"_TOTAL"]) + (y_range*0.1)
    # print("EXIT -------- AxesCallback ---------\n-----------------")

        
def vegMeatTargetSelectCallback(attrname,old,new):
    # print("ENTER -------- TargetCallback ---------")
    veg_meat_params_dict['target_name'] = cat_column_names_dict[veg_meat_target_select.value]
    
    # print("\t\tadded_target_name")
    veg_meat_mult_select.options = list(cat_label_dict[cat_column_names_dict[veg_meat_target_select.value]].values())
    veg_meat_mult_select.value = list(cat_label_dict[cat_column_names_dict[veg_meat_target_select.value]].values())

    # print("\t\tchanged_options")

    veg_meat_prec_rec_val_select.options = list(inv_cat_label_dict[veg_meat_params_dict['target_name']].keys())
    veg_meat_prec_rec_val_select.value = list(inv_cat_label_dict[veg_meat_params_dict['target_name']].keys())[0]
    # print("\t\tchange_prec_rec_options")
    
    vegMeatChangeData()
    vegMeatPrecRecChangeData()
    # print("EXIT -------- TargetCallback ---------\n-----------------")
        

def vegMeatMultSelectCallback(attrname,old,new):
    # print("ENTER -------- MultSelectCallback ---------")
    vegMeatChangeData()
    # print("EXIT -------- MultSelectCallback ---------\n-----------------")
        
        

def vegMeatPrecRecValCallback(attrname,old,new):
    # print("ENTER -------- PrecRecCallback ---------")
    vegMeatPrecRecChangeData()
    # print("EXIT -------- PrecRecCallback ---------\n-----------------")
        
    
veg_meat_x_select.on_change("value",vegMeatAxesCallback)
veg_meat_y_select.on_change("value",vegMeatAxesCallback)
veg_meat_target_select.on_change("value",vegMeatTargetSelectCallback)
veg_meat_mult_select.on_change("value",vegMeatMultSelectCallback)
veg_meat_prec_rec_val_select.on_change("value",vegMeatPrecRecValCallback)





temp_x_range = np.max(all_data.loc[:,veg_meat_params_dict['x_name']+"_TOTAL"]) - np.min(all_data.loc[:,veg_meat_params_dict['x_name']+"_TOTAL"])
temp_y_range = np.max(all_data.loc[:,veg_meat_params_dict['y_name']+"_TOTAL"]) - np.min(all_data.loc[:,veg_meat_params_dict['y_name']+"_TOTAL"])

temp_min_x = np.min(all_data.loc[:,veg_meat_params_dict['x_name']+"_TOTAL"]) - (temp_x_range*0.1)
temp_max_x = np.max(all_data.loc[:,veg_meat_params_dict['x_name']+"_TOTAL"]) + (temp_x_range*0.1)
temp_min_y = np.min(all_data.loc[:,veg_meat_params_dict['y_name']+"_TOTAL"]) - (temp_y_range*0.1)
temp_max_y = np.max(all_data.loc[:,veg_meat_params_dict['y_name']+"_TOTAL"]) + (temp_y_range*0.1)

veg_meat_figure = figure(title="Produce-Meat Scatter Plot",
                        plot_width=500,plot_height=450,
                        x_range=[temp_min_x , temp_max_x],
                        y_range=[temp_min_y , temp_max_y])

veg_meat_figure.circle(x='x_data' , y='y_data' ,
                      fill_color='color_data' , line_color='color_data' , fill_alpha=0.2 , line_alpha=0.2 ,
                      size=3 , 
                      source=veg_meat_background_CDS)

veg_meat_figure.circle(x='x_data' , y='y_data' ,
                      fill_color='color_data' , line_color='color_data' , fill_alpha=0.8 ,
                      size='radius_data' , legend='legend_data' ,
                      source=veg_meat_CDS)


veg_meat_prec_rec_figure = figure(title="Produce-Meat Prec-Rec Plot",
                                 plot_width=500,plot_height=450,
                                 x_range=['T-Act','F-Act'],
                                 y_range=['F-Pred','T-Pred'])

veg_meat_prec_rec_figure.quad(top='t_data' , bottom='b_data' , left='l_data' , right='r_data', 
                               fill_color='color_data' , line_color='color_data' , fill_alpha=0.8 ,
                               source=veg_meat_prec_rec_CDS)





curdoc().add_root(column(column(row(cat_cat_figure , cat_cat_prec_rec_figure) , row(cat_cat_options_box , cat_cat_prec_rec_options_box)),
                      column(row(veg_meat_figure , veg_meat_prec_rec_figure) , row(veg_meat_options_box , veg_meat_prec_rec_options_box))))
