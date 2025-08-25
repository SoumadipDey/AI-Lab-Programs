def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif x == b:
        return 1
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

fuzzy_sets = {
    'service': {
        'very_poor': [0, 0, 1.5],
        'poor':      [1, 1.5, 2.5],
        'average':   [2, 3, 4],
        'good':      [3, 3.5, 4.5],
        'excellent': [4, 5, 5]
    },
    'ambience': {
        'very_poor': [0, 0, 1.5],
        'poor':      [1, 1.5, 2.5],
        'average':   [2, 3, 4],
        'good':      [3, 3.5, 4.5],
        'excellent': [4, 5, 5]
    },
    'food_quality': {
        'very_poor': [0, 0, 1.5],
        'poor':      [1, 1.5, 2.5],
        'average':   [2, 3, 4],
        'good':      [3, 3.5, 4.5],
        'excellent': [4, 5, 5]
    }
}

def compute_memberships(x, sets):
    memberships = {}
    for label, (a, b, c) in sets.items():
        memberships[label] = triangular(x, a, b, c)
    return memberships

def compute_fuzzy_rating(service_deg, ambience_deg, food_deg):
    rating_deg = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    labels_to_stars = {'very_poor':1, 'poor':2, 'average':3, 'good':4, 'excellent':5}
    
    for label, star in labels_to_stars.items():
        
        # Combine service, ambience, and food quality memberships and take their weighted sum
        combined = 0.25*service_deg.get(label,0) + 0.25*ambience_deg.get(label,0) + 0.5*food_deg.get(label,0)
        
        rating_deg[star] = max(rating_deg[star], combined)
    
    return rating_deg

def defuzzify(rating_deg):
    numerator = sum(star * deg for star, deg in rating_deg.items()) # Sum of product of the member and its membership value
    denominator = sum(deg for deg in rating_deg.values()) # Sum of the membership values
    return numerator / denominator if denominator != 0 else 0

def fuzzy_restaurant_rating(inputs):
    service_input, ambience_input, food_input = inputs
    
    service_deg = compute_memberships(service_input, fuzzy_sets['service'])
    ambience_deg = compute_memberships(ambience_input, fuzzy_sets['ambience'])
    food_deg = compute_memberships(food_input, fuzzy_sets['food_quality'])
    
    rating_deg = compute_fuzzy_rating(service_deg, ambience_deg, food_deg)
    
    final_rating = defuzzify(rating_deg)
    return final_rating

inputs = [(4.2, 4.0, 4.5), # Service, Ambience, Food Quality
          (2.4, 2.3, 4.9),
          (3.4, 2.7, 3.8),
          (5.0, 5.0, 2.7)]
  
ratings = [fuzzy_restaurant_rating(value) for value in inputs] 
for curr_input, rating in zip(inputs,ratings):
    service, ambience, food_qual = curr_input
    print(f"When Service = {service}, Ambience = {ambience}, Food Quality = {food_qual}")
    print(f"Predicted Restaurant Rating: {rating:.2f} stars\n")