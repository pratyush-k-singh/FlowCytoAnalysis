import pandas as pd

def user_feedback(data, scores, predictions):
    """
    Collect user feedback on detected anomalies.

    Args:
        data (DataFrame): The original data.
        scores (ndarray): The anomaly scores.
        predictions (ndarray): The anomaly predictions.

    Returns:
        DataFrame: Updated data with user feedback.
    """
    feedback_data = pd.DataFrame({'Score': scores, 'Prediction': predictions})
    feedback_data['User_Label'] = feedback_data['Prediction'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    
    for index, row in feedback_data.iterrows():
        print(f"Data Point:\n{data.iloc[index]}")
        user_input = input(f"Is this point an anomaly? (Y/N): ").strip().upper()
        feedback_data.at[index, 'User_Label'] = 'Anomaly' if user_input == 'Y' else 'Normal'
    
    return feedback_data

def retrain_model(model, data, feedback_data):
    """
    Retrain the IsolationForest model based on user feedback.

    Args:
        model (IsolationForest): The original IsolationForest model.
        data (DataFrame): The original data.
        feedback_data (DataFrame): The user feedback data.

    Returns:
        IsolationForest: The retrained model.
    """
    new_data = data.copy()
    new_data['User_Label'] = feedback_data['User_Label']
    anomalies = new_data[new_data['User_Label'] == 'Anomaly']
    non_anomalies = new_data[new_data['User_Label'] == 'Normal']
    retrain_data = pd.concat([anomalies, non_anomalies], ignore_index=True).drop(columns=['User_Label'])
    model.fit(retrain_data)
    return model
