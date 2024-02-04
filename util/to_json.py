import pandas as pd

data =pd.read_excel('data.xlsx').to_numpy()
# {
#   'name': "Donald Trump",
#   'category': "Half true",
#   'value': 0.13761467889908258
#   }
for i in range(len(data)):
    print('{')
    print('name:\"'+str(data[i][0])+'\",')
    print('category:\"'+str(data[i][1])+'\",')
    print('value:\"'+str(data[i][2])+'\",')
    print('},')
# with open('output.txt' , 'w') as f:
#     f.write()