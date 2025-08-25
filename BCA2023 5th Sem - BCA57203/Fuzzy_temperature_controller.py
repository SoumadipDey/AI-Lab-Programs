def membershipFunction(error):
    magnitude = round(min(1.0, abs(error) / 10.0),3)
    
    if error < 0:   # Too hot -> Cooling dominates
        return {
            "Cooling": magnitude,
            "Heating": round((1 - magnitude) * 0.2,3),
            "Neutral": round((1 - magnitude) * 0.8,3)
        }
    elif error > 0:  # Too cold -> Heating dominates
        return {
            "Heating": magnitude,
            "Cooling": round((1 - magnitude) * 0.2,3),
            "Neutral": round((1 - magnitude) * 0.8,3)
        }
    else:  # Perfect temp -> Hold
        return {"Neutral": 1.0, "Heating": 0.0, "Cooling": 0.0}

DEVICE_ACTIONS = {
    "Cooling": ("AC", 100),      # Max cooling capacity
    "Heating": ("Heater", 100),  # Max heating capacity
    "Neutral": ("Nothing", 0)    # No action
}

def fuzzyDecision(target_temp, current_temp):
    error = target_temp - current_temp
    
    #Fuzzification
    memberships = membershipFunction(error)

    #De-fuzzification using maximum membership method
    strongest_label = max(memberships, key=memberships.get)
    strength = memberships[strongest_label]    
    device, max_power = DEVICE_ACTIONS[strongest_label]
    power = round(strength * max_power, 1)  # scale with membership degree

    # Print useful info
    print(f"Target Temp: {target_temp}, Current Temp: {current_temp}")
    print(f"Error: {error}")
    print("Fuzzy memberships:", memberships)
    print(f"{device} ON at {power}% power")

target_temps = [20, 20, 30, -20, 23, 20] # In Degree C
current_temps = [28, 12, 25, 15, 32, 23] # In Degree C

print("Testing the fuzzy controller for a variety of situations:")
for target, curr in zip(target_temps,current_temps):
    fuzzyDecision(target,curr)
    print("---------------------------------")