import pandas as pd 
from tqdm import tqdm
import os
import pandas as pd 
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

def feature_concatenation(x, y):
        return np.concatenate((x, y), axis=-1)

def feature_add_subtract(x, y):
    return np.concatenate((x + y, x - y), axis=-1)

def feature_add_subtract_multiply(x, y):
    return np.concatenate((x + y, x - y, x * y), axis=-1)

def feature_squared_difference_squared_sum(x, y):
    return np.concatenate((x**2 - y**2, (x - y)**2), axis=-1)

def feature_squared_difference_squared_sum_multiply(x, y):
    return np.concatenate((x**2 - y**2, (x - y)**2, x * y), axis=-1)

def cal_feature(p_path, model):
    image = imread(p_path)
    image_224 = resize(image, (224, 224), preserve_range=True, mode='reflect')
    image_224_batch = np.expand_dims(image_224, axis=0)
    preprocessed_batch = preprocess_input(image_224_batch)
    feature_arr = model.predict(preprocessed_batch)
    return feature_arr

def fuse_features(feature_x, feature_y, fusion_type='concatenation'):
    if fusion_type == 'concatenation':
        return feature_concatenation(feature_x, feature_y)
    elif fusion_type == 'add_subtract':
        return feature_add_subtract(feature_x, feature_y)
    elif fusion_type == 'add_subtract_multiply':
        return feature_add_subtract_multiply(feature_x, feature_y)
    elif fusion_type == 'squared_difference_squared_sum':
        return feature_squared_difference_squared_sum(feature_x, feature_y)
    elif fusion_type == 'squared_difference_squared_sum_multiply':
        return feature_squared_difference_squared_sum_multiply(feature_x, feature_y)
    else:
        raise ValueError("Invalid fusion type")
    


def calculate_similarity(p_path1, p_path2, model, fusion_type='concatenation'):
    feature_x = cal_feature(p_path1, model)
    feature_y = cal_feature(p_path2, model)
    fused_features = fuse_features(feature_x, feature_y, fusion_type=fusion_type)
    similarity_score = compute_similarity(fused_features)
    return similarity_score

def compute_similarity(fused_features):
    fc1 = Dense(128, activation='relu')(fused_features)
    fc2 = Dense(1, activation='sigmoid')(fc1)
    return fc2

def enrich_face_image_pairs(path, df):
    new_pair = pd.DataFrame(columns=['p1_path', 'p2_path', 'ptype', 'tag'])
    index = 0
    tag = 0
    for i in tqdm(range(len(df))):
        p1 = df.iloc[i]['p1']
        p2 = df.iloc[i]['p2']
        ptype = df.iloc[i]['ptype']
        for p1_path in os.listdir(path + p1):
            for p2_path in os.listdir(path + p2):
                new_pair.loc[index] = [p1 + '/' + p1_path, p2 + '/' + p2_path, ptype, tag]
                index += 1
                tag += 1
    return new_pair

def create_base_network(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
    return base_model

def get_full_relationship_name(abbreviation):
    relationship_mapping = {
        'bb': 'brother-brother',
        'ss': 'sister-sister',
        'sibs': 'sister-brother',
        'fd': 'father-daughter',
        'fs': 'father-son',
        'md': 'mother-daughter',
        'ms': 'mother-son',
        'gfgd': 'grandfather-granddaughter',
        'gfgs': 'grandfather-grandson',
        'gmgd': 'grandmother-granddaughter',
        'gmgs': 'grandmother-grandson'
    }
    
    return relationship_mapping.get(abbreviation.lower(), "Unknown relationship")

def switch_label(label):
    return{
        'ms':0,
        'fs':1,
        'bb':2,
        'sibs':3,
        'fd':4,
        'md':5,
        'ss':6,
        'gfgs':7,
        'gfgd':8,
        'gmgs':9,
        'gmgd':10
    }.get(label)

def pre2label(pred):
    index = np.argmax(pred)
    confidence = pred[index]
    label = {
        0: 'ms',
        1: 'fs',
        2: 'bb',
        3: 'sibs',
        4: 'fd',
        5: 'md',
        6: 'ss',
        7: 'gfgs',
        8: 'gfgd',
        9: 'gmgs',
        10: 'gmgd'
    }.get(index)
    return label, confidence

def get_pretrained_model(model_filename, fusion_input_dim):
    repo_name = 'Leo1212/DSPRO2'

    model_weights_path = hf_hub_download(repo_id=repo_name, filename=model_filename)

    # Assume `fusion_input_dim` and `output_dim` are defined as in your previous code
    output_dim = 7  # Number of classes

    # Define the model architecture consistent with the training phase
    def build_model(input_dim, output_dim):
        model = Sequential()
        model.add(Dense(100, activation='relu', input_dim=input_dim))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(output_dim, activation='sigmoid'))
        return model

    # Build the model
    model = build_model(fusion_input_dim, output_dim)

    # Load the model weights
    model.load_weights(model_weights_path)

    # Compile the model (optional, if you want to use it for further training or evaluation)
    sgd = SGD(learning_rate=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def fromDataset(test_data_path, test_excel_path, fusion_type, base_model, pretrained_model):

    train_faces_path=test_data_path

    ignore_ptypes = ['gmgs', 'gfgs', 'gfgd', 'gmgd']

    df=pd.read_excel(test_excel_path)
    df=df[['p1','p2','ptype']]

    df = df[~df['ptype'].isin(ignore_ptypes)]

    new_pair = enrich_face_image_pairs(train_faces_path, df)
    tqdm.pandas(desc='Processing:')
    new_pair['p1_path']=new_pair.progress_apply(lambda x: train_faces_path+x['p1_path'], axis=1)
    new_pair['p2_path']=new_pair.progress_apply(lambda x: train_faces_path+x['p2_path'], axis=1)
    
    
    lis=[]
    for f in os.listdir(train_faces_path):
        for mid in os.listdir(train_faces_path+f):
            if 'MID' in mid:
                for image in os.listdir(train_faces_path+f+'/'+mid):
                    lis.append(train_faces_path+f+'/'+mid+'/'+image)

    dic={}
    for image in tqdm(lis):
        dic[image]=cal_feature(image, base_model)

    tqdm.pandas(desc='Processing:')
    new_pair['p1_feature'] = new_pair.apply(lambda x: dic[x['p1_path']], axis=1)
    new_pair['p2_feature'] = new_pair.apply(lambda x: dic[x['p2_path']], axis=1)

    new_pair['feature_distance'] = new_pair.apply(lambda x: fuse_features(x['p1_feature'], x['p2_feature'], fusion_type), axis=1)

    ptype_arr=new_pair['ptype'].values
    distance=new_pair['feature_distance'].values

    l=[]
    # Loop through the data in chunks and stack them vertically
    for i in tqdm(range(60)):
        chunk = distance[i*4409:(i+1)*4409]
        if len(chunk) > 0:  # Check if the chunk is not empty
            dis_arr = np.vstack(chunk)
            l.append(dis_arr)

    # Combine all sub-arrays into one if the list is not empty
    if l:
        dis_arr = np.vstack(l)
    else:
        dis_arr = np.array([])  # Handle the case where no arrays were added

    preds = pretrained_model.predict(dis_arr)

    preds_label=np.array(list(map(pre2label,preds)))
    return accuracy_score(ptype_arr, preds_label) 

def predict_label(image1_path, image2_path, base_model, pretrained_model, fusion_type='concatenation'):
    feature_x = cal_feature(image1_path, base_model)
    feature_y = cal_feature(image2_path, base_model)
    fused_features = fuse_features(feature_x, feature_y, fusion_type=fusion_type)
    
    # Flatten fused features if necessary
    fused_features_flat = fused_features.flatten().reshape(1, -1)
    
    preds = pretrained_model.predict(fused_features_flat)
    return pre2label(preds[0])

# possible values: (concatenation, add_subtract, add_subtract_multiply, squared_difference_squared_sum, squared_difference_squared_sum_multiply)
fusion_type = 'concatenation'

# possible values: (concatenation: 4096, add_subtract: 4096, add_subtract_multiply: 6144, squared_difference_squared_sum: 4096, squared_difference_squared_sum_multiply: 6144)
fusion_input_dim = 4096


base_model = create_base_network((224, 224, 3))
model = get_pretrained_model('concatenation_model-best.h5', fusion_input_dim)

def predict(image1_path, image2_path):
    return predict_label(image1_path, image2_path, base_model, model, fusion_type)




# predicted_label, confidence  = predict_label('train-faces/F0001/MID1/P00001_face0.jpg', 'train-faces/F0001/MID3/P00006_face0.jpg', base_model, model, fusion_type)
# print(f'Predicted label: {predicted_label}, Confidence: {confidence*100:.2f}%')

# test_accuracy = fromDataset('data/Family101/', 'data/test_family101.xlsx', fusion_type, base_model, model)
# print(f'Test accuracy: {test_accuracy}')