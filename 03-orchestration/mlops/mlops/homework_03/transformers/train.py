import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):

    onehotDF = pd.DataFrame()

    onehotDF['PULocationID'] = data['PULocationID'].astype(str)
    onehotDF['DOLocationID'] = data['DOLocationID'].astype(str)
    onehotDF['duration'] = data['duration']
    onehotDF = onehotDF.drop_duplicates()

    #onehotDFDict = onehotDF[['PULocationID','DOLocationID']].drop_duplicates().to_dict(orient='records')
    dataDicts = onehotDF[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    target = onehotDF['duration'].values

    dicVec = DictVectorizer(sparse=False)
    #feature_matrix = dicVec.fit_transform(onehotDFDict)
    feature_matrix = dicVec.fit_transform(dataDicts)
    n_feature_cols = feature_matrix.shape[1]

    # Define the chunk size
    chunk_size = 100  # You can adjust this according to your dataset size

    # Initialize the model
    model = LinearRegression()

    # Initialize lists to store predictions and targets
    all_predictions = []
    all_targets = []

    # Iterate over the data in chunks
    for i in range(0, len(feature_matrix), chunk_size):
        # Get the chunk of features and targets
        X_chunk = feature_matrix[i:i+chunk_size]
        y_chunk = target[i:i+chunk_size]

        # Train the model on the chunk
        model.fit(X_chunk, y_chunk)

    for i in range(0, len(feature_matrix), chunk_size):
        # Get the chunk of features and targets
        X_chunk = feature_matrix[i:i+chunk_size]
        y_chunk = target[i:i+chunk_size]
        
        # Predict on the chunk
        chunk_predictions = model.predict(X_chunk)
        y_chunk = target[i:i+chunk_size]

        # Store predictions and targets
        all_predictions.extend(chunk_predictions)
        all_targets.extend(y_chunk)

    print(model.intercept_)

    return model, dicVec


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'