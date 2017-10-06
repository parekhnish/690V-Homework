import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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



def makeDataDictForCatCat(params_dict,visible_labels,all_data,color_dict,label_dict):

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
            count_mat[:,:,i] = count_mat[:,:,i] / num_rows

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

    # ret_dict['x_data'] = [label_dict[x_name][i] for i in list(x_data)]
    # ret_dict['y_data'] = [label_dict[y_name][i] for i in list(y_data)]
    ret_dict['x_data'] = [i+0.5 for i in list(x_data)]
    ret_dict['y_data'] = [i+0.5 for i in list(y_data)]
    ret_dict['target_data'] = target_data
    ret_dict['radius_data'] = radius_data
    ret_dict['legend_data'] = [label_dict[target_name][i] for i in list(ret_dict['target_data'])]
    ret_dict['color_data'] = [color_dict[i] for i in list(ret_dict['target_data'])]

    return ret_dict



def makeDataDictForBackgroundCatCat(params_dict,tree,all_data,color_dict,label_dict):

    num_rows = len(list(all_data.index))

    x_name = params_dict['x_name']
    y_name = params_dict['y_name']

    num_x_values = len(label_dict[x_name].keys())
    num_y_values = len(label_dict[y_name].keys())

    x_coords = np.linspace(0,num_x_values,num=num_x_values*50)
    y_coords = np.linspace(0,num_y_values,num=num_y_values*50)

    num_x_coords = x_coords.shape[0]
    num_y_coords = y_coords.shape[0]

    x_coords = np.tile(x_coords[...,None] , (1,num_y_coords))
    y_coords = np.tile(y_coords[None,...] , (num_x_coords,1))

    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    features = np.stack([x_coords,y_coords],axis=1)
    labels = tree.predict(features)

    ret_dict = {}
    ret_dict['x_data'] = x_coords
    ret_dict['y_data'] = y_coords
    ret_dict['color_data'] = [color_dict[i] for i in list(labels)]

    return ret_dict




def makeDataDictForCatCatPrecRec(params_dict,tree,curr_id,all_data,color_dict,label_dict):

    curr_id = int(curr_id)
    num_rows = len(list(all_data.index))

    target_name = params_dict['target_name']

    corr_labels = all_data.loc[:,target_name]
    pred_labels = getPredLabelsForCatCat(params_dict,tree,all_data,label_dict)

    size_data = np.empty((4), dtype=np.float64)

    is_corr = (corr_labels == curr_id)
    is_not_corr = np.invert(is_corr)
    is_pred = (pred_labels == curr_id)
    is_not_pred = np.invert(is_pred)

    size_data[0] = np.sum(np.logical_and(is_pred,is_corr))
    size_data[1] = np.sum(np.logical_and(is_pred,is_not_corr))
    size_data[2] = np.sum(np.logical_and(is_not_pred,is_corr))
    size_data[3] = np.sum(np.logical_and(is_not_pred,is_not_corr))

    size_data = size_data / float(num_rows)

    # x_data = ['T-Act','F-Act','T-Act','F-Act']
    x_data = [0.5,1.5,0.5,1.5]
    # y_data = ['T-Pred','T-Pred','F-Pred','F-Pred']
    y_data = [1.5,1.5,0.5,0.5]
    color_data = [color_dict[2] , color_dict[3] , color_dict[3] , color_dict[2]]

    ret_dict = {}
    ret_dict['t_data'] = y_data + size_data/2.0
    ret_dict['b_data'] = y_data - size_data/2.0
    ret_dict['l_data'] = x_data - size_data/2.0
    ret_dict['r_data'] = x_data + size_data/2.0
    ret_dict['color_data'] = color_data

    # print("CURR ID: " + str(curr_id))
    # print(np.stack((corr_labels,pred_labels),axis=1))
    # print("===========================")
    # print("===========================")

    return ret_dict



def makeTreeForCatCat(params_dict,all_data,label_dict):

    num_rows = len(list(all_data.index))

    x_name = params_dict['x_name']
    y_name = params_dict['y_name']
    target_name = params_dict['target_name']

    num_x_values = len(label_dict[x_name].keys())
    num_y_values = len(label_dict[y_name].keys())
    num_target_values = len(label_dict[target_name].keys())

    features = (all_data.loc[:,[x_name,y_name]]).values
    features = np.concatenate((features + 0.8 , features + 0.2),axis=0)
    labels = (all_data.loc[:,target_name]).values
    labels = np.concatenate((labels,labels),axis=0)

    tree = DecisionTreeClassifier(max_depth=3).fit(features,labels)
    # print(np.stack((labels,tree.predict(features)),axis=1))
    # print("===========================")

    return tree


def getPredLabelsForCatCat(params_dict,tree,all_data,label_dict):

    num_rows = len(list(all_data.index))

    x_name = params_dict['x_name']
    y_name = params_dict['y_name']
    target_name = params_dict['target_name']

    num_x_values = len(label_dict[x_name].keys())
    num_y_values = len(label_dict[y_name].keys())
    num_target_values = len(label_dict[target_name].keys())

    features = (all_data.loc[:,[x_name,y_name]]).values + 0.5

    pred_labels = tree.predict(features)
    
    return pred_labels





# BOOL-Pet , BOOL-Historical Smoking , BOOL-Diabetes
# TYPE-Hand , BOOL-Diabetes , BOOL-Rash
# --------------------------------
# --------------------------------
# --------------------------------
# --------------------------------
# --------------------------------

def makeDataDictForVegMeat(params_dict,visible_labels,all_data,color_dict,label_dict):

    num_rows = len(list(all_data.index))

    x_name = params_dict['x_name']+"_TOTAL"
    y_name = params_dict['y_name']+"_TOTAL"
    target_name = params_dict['target_name']
    radius_scale = params_dict['radius_scale']

    x_data = all_data.loc[:,x_name]
    y_data = all_data.loc[:,y_name]
    target_data = all_data.loc[:,target_name]

    is_visible = np.array([ label_dict[target_name][x] in visible_labels for x in target_data])

    x_data = x_data[is_visible]
    y_data = y_data[is_visible]
    target_data = target_data[is_visible]

    radius_data = np.ones((x_data.shape[0]) , dtype=np.float64) * radius_scale


    ret_dict = {}

    ret_dict['x_data'] = x_data
    ret_dict['y_data'] = y_data
    ret_dict['target_data'] = target_data
    ret_dict['radius_data'] = radius_data
    ret_dict['legend_data'] = [label_dict[target_name][i] for i in list(ret_dict['target_data'])]
    ret_dict['color_data'] = [color_dict[i] for i in list(ret_dict['target_data'])]

    return ret_dict



def makeDataDictForBackgroundVegMeat(params_dict,tree,all_data,color_dict):

    num_rows = len(list(all_data.index))

    x_name = params_dict['x_name']+"_TOTAL"
    y_name = params_dict['y_name']+"_TOTAL"

    x_range = np.max(all_data.loc[:,x_name]) - np.min(all_data.loc[:,x_name])
    y_range = np.max(all_data.loc[:,y_name]) - np.min(all_data.loc[:,y_name])

    x_min = np.min(all_data.loc[:,x_name]) - (x_range*0.1)
    x_max = np.max(all_data.loc[:,x_name]) + (x_range*0.1)
    y_min = np.min(all_data.loc[:,y_name]) - (y_range*0.1)
    y_max = np.max(all_data.loc[:,y_name]) + (y_range*0.1)

    x_coords = np.linspace(x_min,x_max,num=100)
    y_coords = np.linspace(y_min,y_max,num=100)

    num_x_coords = x_coords.shape[0]
    num_y_coords = y_coords.shape[0]

    x_coords = np.tile(x_coords[...,None] , (1,num_y_coords))
    y_coords = np.tile(y_coords[None,...] , (num_x_coords,1))

    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    features = np.stack([x_coords,y_coords],axis=1)
    labels = tree.predict(features)

    ret_dict = {}
    ret_dict['x_data'] = x_coords
    ret_dict['y_data'] = y_coords
    ret_dict['color_data'] = [color_dict[i] for i in list(labels)]

    return ret_dict



def makeDataDictForVegMeatPrecRec(params_dict,tree,curr_id,all_data,color_dict):

    curr_id = int(curr_id)
    num_rows = len(list(all_data.index))

    target_name = params_dict['target_name']

    corr_labels = all_data.loc[:,target_name]
    pred_labels = getPredLabelsForVegMeat(params_dict,tree,all_data)

    size_data = np.empty((4), dtype=np.float64)

    is_corr = (corr_labels == curr_id)
    is_not_corr = np.invert(is_corr)
    is_pred = (pred_labels == curr_id)
    is_not_pred = np.invert(is_pred)

    size_data[0] = np.sum(np.logical_and(is_pred,is_corr))
    size_data[1] = np.sum(np.logical_and(is_pred,is_not_corr))
    size_data[2] = np.sum(np.logical_and(is_not_pred,is_corr))
    size_data[3] = np.sum(np.logical_and(is_not_pred,is_not_corr))

    size_data = size_data / float(num_rows)

    # x_data = ['T-Act','F-Act','T-Act','F-Act']
    x_data = [0.5,1.5,0.5,1.5]
    # y_data = ['T-Pred','T-Pred','F-Pred','F-Pred']
    y_data = [1.5,1.5,0.5,0.5]
    color_data = [color_dict[2] , color_dict[3] , color_dict[3] , color_dict[2]]

    ret_dict = {}
    ret_dict['t_data'] = y_data + size_data/2.0
    ret_dict['b_data'] = y_data - size_data/2.0
    ret_dict['l_data'] = x_data - size_data/2.0
    ret_dict['r_data'] = x_data + size_data/2.0
    ret_dict['color_data'] = color_data

    # print("CURR ID: " + str(curr_id))
    # print(np.stack((corr_labels,pred_labels),axis=1))
    # print("===========================")
    # print("===========================")

    return ret_dict



def makeTreeForVegMeat(params_dict,all_data):

    num_rows = len(list(all_data.index))

    x_name = params_dict['x_name']+"_TOTAL"
    y_name = params_dict['y_name']+"_TOTAL"
    target_name = params_dict['target_name']

    features = (all_data.loc[:,[x_name,y_name]]).values
    labels = (all_data.loc[:,target_name]).values

    tree = DecisionTreeClassifier(max_depth=3).fit(features,labels)

    return tree



def getPredLabelsForVegMeat(params_dict,tree,all_data):

    num_rows = len(list(all_data.index))

    x_name = params_dict['x_name']+"_TOTAL"
    y_name = params_dict['y_name']+"_TOTAL"
    target_name = params_dict['target_name']

    features = (all_data.loc[:,[x_name,y_name]]).values

    pred_labels = tree.predict(features)
    
    return pred_labels



