import pandas as hsp
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

#Reading Training csv dataset using pandas
try:
    #Taking necessary columns from dataset
    tds = hsp.read_csv("train.csv", usecols=["LotArea", "SalePrice","FullBath","BedroomAbvGr"])
except FileNotFoundError:
    print("File not found.")
    exit()

#Training The model
x = tds[["LotArea","FullBath","BedroomAbvGr"]]
y = tds["SalePrice"]
lr.fit(x,y)


#Creating a new panda frame and testing a new test data and saving it, for larger predictions
ts = hsp.read_csv("test.csv", usecols=["LotArea","FullBath","BedroomAbvGr"])
tl = lr.predict(ts)
ts["SalePrice"] = tl
ts.to_csv('test_result.csv',index=False)