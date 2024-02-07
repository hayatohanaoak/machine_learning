from absentieesm_module import absenteeismModel
from absentieesm_module import CustomScaler

model = absenteeismModel('.\model', '.\scaler')
model.load_and_clean_data('.\Absenteeism_new_data.csv')
model_data = model.predicted_outputs
model_data.to_csv('Absenteeism_prediction.csv', index = False)