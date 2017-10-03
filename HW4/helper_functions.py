import numpy as np
import pandas as pd


def makeBoolColumn(all_data,col_name_list,symb):
    
    num_rows = len(list(all_data.index))
    num_cols = len(col_name_list)
    
    new_col = np.zeros((num_rows),dtype=np.int64)
    for i in range(num_cols):
        
        curr_col = np.array(all_data[col_name_list[i]])
        new_col[curr_col==symb] = 1
        
    return new_col


def makeTypeColumn(all_data,col_name_list,symb,is_zero_inc):
    
    num_rows = len(list(all_data.index))
    num_cols = len(col_name_list)
    
    new_col = np.zeros((num_rows),dtype=np.int64)


    for i in range(num_cols):
        
        curr_col = np.array(all_data[col_name_list[i]])
        if(is_zero_inc):
            new_col[curr_col==symb] = i
        else:
            new_col[curr_col==symb] = i+1
        
    return new_col



def makeDataDictForScatter(params_dict,visible_labels,all_data,color_dict,label_dict):

    num_rows = len(list(all_data.index))

    x_name = params_dict['x_name']
    y_name = params_dict['y_name']
    target_name = params_dict['target_name']
    radius_scale = params_dict['radius_scale']

    num_x_values = len(label_dict[x_name].keys())
    num_y_values = len(label_dict[y_name].keys())
    num_target_values = len(label_dict[target_name].keys())

    target_total = np.zeros((num_target_values) , dtype=np.float64)
    count_mat = np.zeros((num_x_values,num_y_values,num_target_values) , dtype=np.float64)

    raw_values = all_data.loc[:,[x_name,y_name,target_name]]

    for i in range(num_rows):
        if(label_dict[target_name][raw_values.loc[i][2]] in visible_labels):
            count_mat[raw_values.loc[i][0],raw_values.loc[i][1],raw_values.loc[i][2]] += 1
        target_total[raw_values.loc[i][2]] += 1

    for i in range(num_target_values):
        if(target_total[i] > 0):
            count_mat[:,:,i] = count_mat[:,:,i] / target_total[i]

    x_val_mat =         np.tile(np.arange(num_x_values)[...,None,None],(1,num_y_values,num_target_values))
    y_val_mat =         np.tile(np.arange(num_y_values)[None,...,None],(num_x_values,1,num_target_values))
    target_val_mat =    np.tile(np.arange(num_target_values)[None,None,...],(num_x_values,num_y_values,1))

    x_data = x_val_mat.flatten()
    y_data = y_val_mat.flatten()
    target_data = target_val_mat.flatten()
    radius_data = count_mat.flatten() * radius_scale

    is_not_zero = np.invert(radius_data==0)

    x_data = x_data[is_not_zero]
    y_data = y_data[is_not_zero]
    target_data = target_data[is_not_zero]
    radius_data = radius_data[is_not_zero]

    ret_dict = {}

    # ret_dict['x_data'] = x_data[is_not_zero]
    # ret_dict['y_data'] = y_data[is_not_zero]
    ret_dict['x_data'] = [label_dict[x_name][i] for i in list(x_data)]
    ret_dict['y_data'] = [label_dict[y_name][i] for i in list(y_data)]
    ret_dict['target_data'] = target_data
    ret_dict['radius_data'] = radius_data
    ret_dict['legend_data'] = [label_dict[target_name][i] for i in list(ret_dict['target_data'])]
    ret_dict['color_data'] = [color_dict[i] for i in list(ret_dict['target_data'])]

    return ret_dict



# def cat_cat_ticker_func(label_dict):
#     if(((int(tick) - tick) == 0) and int(tick) < len(label_dict.keys())):
#         return str(label_dict[tick])
#     else
#         return ""




# cat_cat_data_dict = makeDataDictForScatter(cat_cat_params_dict , all_data , color_dict, label_dict)
# cat_cat_CDS = ColumnDataSource(data=cat_cat_data_dict)

# cat_cat_figure = figure(title="Category-Category Scatter Plot",
#                         plot_width=500,plot_height=550,
#                         x_range=[0,len(label_dict[cat_cat_dict['x_name']].keys())],
#                         y_range=[0,len(label_dict[cat_cat_dict['y_name']].keys())])

# cat_cat_figure.circle(x='x_data' , y='y_data' ,
#                       fill_colors='color_data' , line_colors='color_data' , fill_alpha=0.8 ,
#                       size='radius_data' , legend='legend_data' ,
#                       source=cat_cat_CDS)

# show(cat_cat_figure, notebook_handle=True)


# cat_cat_params_dict = {
#                     x_name: 'type_smoking',
#                     y_name: 'bool_pet',
#                     target_name: 'type_pisa'
#                     radius_scale: 100
# }


# def cat_cat_x_ticker_func():
#     d = label_dict[cat_cat_params_dict['x_name']]
    
#     if(((int(tick) - tick) == 0) and int(tick) < len(d.keys())):
#         return str(d[tick])
#     else:
#         return ""
    
# def cat_cat_y_ticker_func():
#     d = label_dict[cat_cat_params_dict['y_name']]
    
#     if(((int(tick) - tick) == 0) and int(tick) < len(d.keys())):
#         return str(d[tick])
#     else:
#         return ""


cat_cat_col_name_dict = {
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
