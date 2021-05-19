

#import statements
import pandas as pd

#config
config = {
    'data_file' : '/Users/fabio.casati/Documents/dev/labdev/data/az/',
    'AL_algo' : 'None', # al algo to select
    'label_col' : '',
    'text_feature_col' : '',
    'categorical_features_cols' :[],
    'num_initial_samples': 1000,
    'validation_set_perc': 0.2,
    'batch_size': 500,
}

#code

def run_experiment(config):
    df = pd.read_csv(config['data_file'])
    label = config['label_col']
    features = [config['text_feature_col'] ] + config['categorical_features_cols']
    features_and_label = features + [label]
    df = df[features_and_label].fillna('')
    val_set_size = min (int(len(df)*config['validation_set_perc']  ), 1000) # or any constant
    #split train-test-eval
    total_train_df = df.sample(val_set_size)
    validation_df =  df.drop(total_train_df.index)



    #init
    cumulative_train_df = total_train_df.sample(config['num_initial_samples'])
    remaining_train_df =  total_train_df.drop(cumulative_train_df.index)

    #loop
    num_batches = int (len(total_train_df)/ config['batch_size'])
    for b in range (num_batches):
        model.fit(cumulative_train_df, config) # "model' includes your entire pipe
        model.eval() #
        next_batch_df = alstrategy.get_next_sample(remaining_train_df, model,... any other params)
        cumulative_train_df = pd.concat([cumulative_train_df, next_batch_df])







