import pandas as pd

data = [
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
]
print(pd.DataFrame(data))

# Separate data by COVID outcome
covid_yes = [row for row in data if row["COVID"] == 1]
covid_no = [row for row in data if row["COVID"] == 0]

# Priors
p_covid = len(covid_yes) / len(data)
p_not_covid = len(covid_no) / len(data)

print("Prior P(COVID=1):", p_covid)
print("Prior P(COVID=0):", p_not_covid)


def probability(feature, value, subset):
    count = sum(1 for row in subset if row[feature] == value)
    return count / len(subset) if subset else 0


def predict(sample):

    # Likelihoods given COVID = 1
    p_x_given_covid = (
        probability("Fever", sample["Fever"], covid_yes)
        * probability("Cough", sample["Cough"], covid_yes)
        * probability("Low_Oxygen", sample["Low_Oxygen"], covid_yes)
    )

    # Likelihoods given COVID = 0
    p_x_given_not_covid = (
        probability("Fever", sample["Fever"], covid_no)
        * probability("Cough", sample["Cough"], covid_no)
        * probability("Low_Oxygen", sample["Low_Oxygen"], covid_no)
    )

    # Numerators
    num_covid = p_x_given_covid * p_covid
    num_not_covid = p_x_given_not_covid * p_not_covid

    # Normalization
    if num_covid + num_not_covid == 0:
        return None  # avoid division by zero
    posterior_covid = num_covid / (num_covid + num_not_covid)

    return posterior_covid

# Example test case
test_sample = {"Fever": 1, "Cough": 0, "Low_Oxygen": 0}
print("\nTest sample:", test_sample)
print("Posterior P(COVID=1):", predict(test_sample))