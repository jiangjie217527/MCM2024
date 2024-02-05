import pandas as pd
data =pd.read_excel('output_eta=0.5.xlsx').to_numpy()
index = []
out = []
for i in range(0,26):
    index.append(i)
for i in range(len(data)):
    inter = False
    for j in range(len(index)):
        if index[j] == data[i][0]:
            inter = True
            break
    if inter:
        out.append(data[i])

print(out)
df = pd.DataFrame(out)
df.to_excel('output.xlsx', index=False)