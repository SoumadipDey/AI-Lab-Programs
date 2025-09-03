import pandas as pd

database = [
    {"Fever": 1, "Cough": 1, "Low_Oxygen": 1, "COVID": 1},
    {"Fever": 0, "Cough": 1, "Low_Oxygen": 0, "COVID": 0},
    {"Fever": 1, "Cough": 0, "Low_Oxygen": 0, "COVID": 1},
    {"Fever": 1, "Cough": 1, "Low_Oxygen": 0, "COVID": 1},
    {"Fever": 0, "Cough": 0, "Low_Oxygen": 0, "COVID": 0},
    {"Fever": 1, "Cough": 1, "Low_Oxygen": 1, "COVID": 1},
    {"Fever": 0, "Cough": 1, "Low_Oxygen": 1, "COVID": 1},
    {"Fever": 1, "Cough": 0, "Low_Oxygen": 1, "COVID": 0},
    {"Fever": 0, "Cough": 0, "Low_Oxygen": 1, "COVID": 0},
    {"Fever": 0, "Cough": 1, "Low_Oxygen": 0, "COVID": 1},
    {"Fever": 0, "Cough": 1, "Low_Oxygen": 1, "COVID": 1},
    {"Fever": 1, "Cough": 1, "Low_Oxygen": 1, "COVID": 1},
    {"Fever": 0, "Cough": 0, "Low_Oxygen": 1, "COVID": 0},
    {"Fever": 0, "Cough": 0, "Low_Oxygen": 0, "COVID": 0},
    {"Fever": 1, "Cough": 1, "Low_Oxygen": 0, "COVID": 1}
]

df = pd.DataFrame(database)
print("Our previously saved database of events")
print(df)

# Calculating probablity of each individual features
# Here y is the dependent event and X is the independent event
def calcConditionalProbablity(Y_set:pd.DataFrame, X:str, valueX:int):
  return Y_set[X].value_counts()[valueX]/len(Y_set)

def predictCovidYES(sample:dict, df:pd.DataFrame) -> float:
  COVID_YES_SET = df[df["COVID"] == 1]
  COVID_NO_SET = df[df["COVID"] == 0]

  # Calculating P(COVID) and P(¬COVID)
  P_COVID_YES = len(COVID_YES_SET) / len(df)
  P_COVID_NO =  len(COVID_NO_SET) / len(df)

  # Calculating P(X1,X2,...Xn|COVID=1)
  P_X_COVID_YES = 1
  for feature in sample.keys():
    P_X_COVID_YES *= calcConditionalProbablity(COVID_YES_SET, feature, sample[feature])

  # Calculating P(X1,X2,...Xn|COVID=0)
  P_X_COVID_NO = 1
  for feature in sample.keys():
    P_X_COVID_NO *= calcConditionalProbablity(COVID_NO_SET, feature, sample[feature])

  # Calculating P(X1,X2,...,Xn) = P(X1,X2,...,Xn|COVID=1)xP(COVID=1) + P(X1,X2,...,Xn|COVID=0)xP(COVID=0)
  P_X_ALL_CLASSES = P_X_COVID_YES*P_COVID_YES + P_X_COVID_NO*P_COVID_NO

  # Implementing Generalized Bayes Rule
  # P(COVID=1|X1,X2,...,Xn) = (P(X1,X2,...,Xn|COVID=1)xP(COVID=1))/P(X1,X2,...,Xn)
  return (P_X_COVID_YES*P_COVID_YES)/P_X_ALL_CLASSES

# Driver Code
test_sample = {"Fever": 1, "Cough": 1, "Low_Oxygen": 1}
print("\nTest sample:", test_sample)
print("Posterior P(COVID=1):", predictCovidYES(test_sample,df))

if(predictCovidYES(test_sample,df)>=0.5):
  print("It is very likely this person has COVID!")
else:
  print("It is likely this person does not have COVID!")