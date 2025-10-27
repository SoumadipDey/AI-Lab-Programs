def askQuestion(question:str):
    answer = input(question)
    return answer.lower() == "yes" or answer.lower() == "y"

print("Answer all questions with \"Yes/No\" or \"Y/N\"\n")
fever = askQuestion("1. Do you have high fever?")
headache = askQuestion("2. Do you have severe headache?")
muscle_pain = askQuestion("3. Do you have muscle or joint pain?")
rash = askQuestion("4. Do you have skin rashes?")
nausea = askQuestion("5. Do you feel nausea or vomiting?")
bleeding = askQuestion("6. Have you noticed any bleeding (nose/gums)?")
abdominal_pain = askQuestion("7. Do you have abdominal pain?")
fatigue = askQuestion("8. Are you feeling very weak or tired?")
low_platelets = askQuestion("9. Has your platelet count dropped?")
restlessness = askQuestion("10. Are you feeling restless or irritable?")

# Rule-based logic for Diagnosis
if fever and headache and muscle_pain:
    if bleeding or (low_platelets and abdominal_pain):
        print("\nDiagnosis: Severe Dengue")
        print("Seek immediate medical care!")
    elif rash or nausea or fatigue:
        print("\nDiagnosis: Moderate Dengue")
        print("Stay hydrated and monitor symptoms closely.")
    else:
        print("\nDiagnosis: Mild Dengue")
        print("Rest and consult a doctor if symptoms worsen.")
else:
    print("\nDiagnosis: No clear signs of Dengue.")
    print("If symptoms persist, visit a doctor for confirmation.")
print("\n--- End of Diagnosis ---")