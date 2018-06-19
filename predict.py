import os
import segmentation
from sklearn.externals import joblib

current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
model = joblib.load(model_dir)

cresult = []

for char in segmentation.characters:
    char = char.reshape(1, -1)
    result = model.predict(char)
    cresult.append(result)

print(cresult)

plate_string = ''
for predict in cresult:
    plate_string += predict[0]

print(plate_string)

column_list_copy = segmentation.column_list[:]
segmentation.column_list.sort()
rightplate_string = ''

for each in segmentation.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print(rightplate_string)
