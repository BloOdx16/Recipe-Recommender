import numpy as np
# We assume this file is in the same directory as advanced_recommender.py
# so we can import the trained model and helper functions.
from advanced_recommender import model, calculate_similarity

# --- 1. Recipe Database ---
# In a real application, this would come from a database.
RECIPES = [
    # Filipino
    {'name': 'Classic Chicken Adobo', 'cuisine': 'Filipino', 'ingredients': ['Chicken', 'Soy Sauce', 'Vinegar (Cane)', 'Garlic', 'Bay Leaf', 'Black Pepper']},
    {'name': 'Pork Sinigang', 'cuisine': 'Filipino', 'ingredients': ['Pork', 'Tamarind', 'Fish Sauce', 'Tomato', 'Onion', 'Spinach']},
    {'name': 'Kare-Kare', 'cuisine': 'Filipino', 'ingredients': ['Oxtail', 'Peanut Butter', 'Eggplant', 'Green Beans', 'Shrimp Paste']},
    {'name': 'Lechon Kawali', 'cuisine': 'Filipino', 'ingredients': ['Pork', 'Bay Leaf', 'Black Pepper', 'Fish Sauce']},
    # Korean
    {'name': 'Spicy Gochujang Pork', 'cuisine': 'Korean', 'ingredients': ['Pork', 'Gochujang', 'Soy Sauce', 'Sesame Oil', 'Garlic', 'Ginger']},
    {'name': 'Kimchi Fried Rice', 'cuisine': 'Korean', 'ingredients': ['Kimchi', 'Rice', 'Pork', 'Sesame Oil', 'Gochujang', 'Scallion']},
    {'name': 'Beef Bulgogi', 'cuisine': 'Korean', 'ingredients': ['Beef', 'Soy Sauce', 'Pear', 'Garlic', 'Sesame Oil', 'Scallion']},
    {'name': 'Japchae (Glass Noodle Stir Fry)', 'cuisine': 'Korean', 'ingredients': ['Glass Noodles', 'Beef', 'Spinach', 'Carrot', 'Onion', 'Soy Sauce', 'Sesame Oil']},
    {'name': 'Sundubu Jjigae (Soft Tofu Stew)', 'cuisine': 'Korean', 'ingredients': ['Tofu', 'Kimchi', 'Pork', 'Gochujang', 'Egg', 'Scallion']},
    # Italian
    {'name': 'Creamy Tomato Basil Pasta', 'cuisine': 'Italian-American', 'ingredients': ['Tomato', 'Basil', 'Garlic', 'Onion', 'Cream', 'Parmesan']},
    {'name': 'Spaghetti Carbonara', 'cuisine': 'Italian', 'ingredients': ['Pancetta', 'Egg', 'Parmesan', 'Black Pepper', 'Spaghetti']},
    {'name': 'Margherita Pizza', 'cuisine': 'Italian', 'ingredients': ['Tomato', 'Mozzarella', 'Basil', 'Olive Oil', 'Dough']},
    {'name': 'Mushroom Risotto', 'cuisine': 'Italian', 'ingredients': ['Arborio Rice', 'Mushroom', 'Onion', 'Parmesan', 'White Wine', 'Thyme']},
    {'name': 'Pesto Pasta', 'cuisine': 'Italian', 'ingredients': ['Basil', 'Pine Nuts', 'Garlic', 'Parmesan', 'Olive Oil', 'Pasta']},
    {'name': 'Lasagna Bolognese', 'cuisine': 'Italian', 'ingredients': ['Beef', 'Pork', 'Tomato', 'Onion', 'Carrot', 'Red Wine', 'Parmesan', 'Mozzarella']},
    {'name': 'Cacio e Pepe', 'cuisine': 'Italian', 'ingredients': ['Spaghetti', 'Pecorino Cheese', 'Black Pepper']},
    {'name': 'Osso Buco', 'cuisine': 'Italian', 'ingredients': ['Veal Shank', 'White Wine', 'Carrot', 'Onion', 'Celery', 'Tomato', 'Lemon']},
    {'name': 'Arancini', 'cuisine': 'Italian', 'ingredients': ['Arborio Rice', 'Saffron', 'Mozzarella', 'Beef', 'Peas', 'Breadcrumbs']},
    # Indian
    {'name': 'Indian Lentil Curry (Dal)', 'cuisine': 'Indian', 'ingredients': ['Lentils', 'Onion', 'Garlic', 'Ginger', 'Turmeric', 'Cumin', 'Tomato']},
    {'name': 'Chicken Tikka Masala', 'cuisine': 'Indian', 'ingredients': ['Chicken', 'Yogurt', 'Tomato', 'Garam Masala', 'Ginger', 'Garlic', 'Cream']},
    {'name': 'Palak Paneer', 'cuisine': 'Indian', 'ingredients': ['Spinach', 'Paneer', 'Onion', 'Garlic', 'Garam Masala', 'Turmeric']},
    {'name': 'Chole (Chickpea Curry)', 'cuisine': 'Indian', 'ingredients': ['Chickpeas', 'Onion', 'Tomato', 'Ginger', 'Garlic', 'Coriander', 'Cumin']},
    {'name': 'Butter Chicken', 'cuisine': 'Indian', 'ingredients': ['Chicken', 'Tomato', 'Butter', 'Cream', 'Garam Masala', 'Ginger', 'Garlic']},
    {'name': 'Saag Aloo', 'cuisine': 'Indian', 'ingredients': ['Spinach', 'Potato', 'Onion', 'Turmeric', 'Cumin', 'Ginger']},
    {'name': 'Rogan Josh', 'cuisine': 'Indian', 'ingredients': ['Lamb', 'Yogurt', 'Onion', 'Garlic', 'Ginger', 'Cardamom', 'Cinnamon', 'Clove']},
    {'name': 'Biryani', 'cuisine': 'Indian', 'ingredients': ['Rice', 'Chicken', 'Yogurt', 'Onion', 'Saffron', 'Mint', 'Coriander', 'Garam Masala']},
    {'name': 'Masala Dosa', 'cuisine': 'Indian', 'ingredients': ['Rice', 'Lentils', 'Potato', 'Onion', 'Mustard Seed', 'Turmeric']},
    # Mexican
    {'name': 'Mexican Chicken Tacos', 'cuisine': 'Mexican', 'ingredients': ['Chicken', 'Corn', 'Lime', 'Cilantro', 'Onion', 'Chili']},
    {'name': 'Beef Enchiladas', 'cuisine': 'Mexican', 'ingredients': ['Beef', 'Chili', 'Onion', 'Garlic', 'Cheese (Cheddar)', 'Corn Tortilla']},
    {'name': 'Guacamole', 'cuisine': 'Mexican', 'ingredients': ['Avocado', 'Lime', 'Cilantro', 'Onion', 'Jalapeno', 'Tomato']},
    {'name': 'Carnitas', 'cuisine': 'Mexican', 'ingredients': ['Pork', 'Orange', 'Onion', 'Garlic', 'Cumin', 'Oregano']},
    {'name': 'Ceviche', 'cuisine': 'Mexican', 'ingredients': ['Fish', 'Lime', 'Onion', 'Cilantro', 'Jalapeno', 'Avocado']},
    {'name': 'Pozole Rojo', 'cuisine': 'Mexican', 'ingredients': ['Pork', 'Hominy', 'Chili', 'Onion', 'Garlic', 'Oregano']},
    {'name': 'Mole Poblano', 'cuisine': 'Mexican', 'ingredients': ['Chicken', 'Chili', 'Chocolate', 'Tomato', 'Onion', 'Peanut', 'Cinnamon']},
    # Japanese
    {'name': 'Japanese Miso Salmon', 'cuisine': 'Japanese', 'ingredients': ['Salmon', 'Miso', 'Mirin', 'Soy Sauce', 'Ginger', 'Scallion']},
    {'name': 'Chicken Katsu Curry', 'cuisine': 'Japanese', 'ingredients': ['Chicken', 'Panko', 'Onion', 'Carrot', 'Potato', 'Curry Powder', 'Soy Sauce']},
    {'name': 'Tonkotsu Ramen', 'cuisine': 'Japanese', 'ingredients': ['Pork', 'Noodles', 'Dashi', 'Soy Sauce', 'Scallion', 'Ginger']},
    {'name': 'Sushi (Nigiri)', 'cuisine': 'Japanese', 'ingredients': ['Rice', 'Fish', 'Wasabi', 'Soy Sauce']},
    {'name': 'Teriyaki Chicken', 'cuisine': 'Japanese', 'ingredients': ['Chicken', 'Soy Sauce', 'Mirin', 'Sake', 'Sugar', 'Ginger']},
    {'name': 'Okonomiyaki', 'cuisine': 'Japanese', 'ingredients': ['Cabbage', 'Flour', 'Egg', 'Pork', 'Okonomiyaki Sauce', 'Mayonnaise', 'Nori']},
    {'name': 'Gyudon (Beef Bowl)', 'cuisine': 'Japanese', 'ingredients': ['Beef', 'Onion', 'Soy Sauce', 'Mirin', 'Dashi', 'Rice']},
    # Sri Lankan
    {'name': 'Sri Lankan Chicken Curry', 'cuisine': 'Sri Lankan', 'ingredients': ['Chicken', 'Coconut Milk', 'Onion', 'Garlic', 'Cinnamon', 'Curry Leaves', 'Turmeric']},
    {'name': 'Dhal Curry', 'cuisine': 'Sri Lankan', 'ingredients': ['Lentils', 'Coconut Milk', 'Onion', 'Curry Leaves', 'Mustard Seed', 'Turmeric']},
    # Western / American
    {'name': 'Rosemary Roasted Beef', 'cuisine': 'Western', 'ingredients': ['Beef', 'Rosemary', 'Thyme', 'Garlic', 'Onion', 'Potato']},
    {'name': 'Classic Cheeseburger', 'cuisine': 'American', 'ingredients': ['Beef', 'Cheese (Cheddar)', 'Lettuce', 'Tomato', 'Onion', 'Bun']},
    {'name': 'BBQ Pulled Pork', 'cuisine': 'American', 'ingredients': ['Pork', 'BBQ Sauce', 'Onion', 'Vinegar (Apple Cider)', 'Brown Sugar']},
    {'name': 'Macaroni and Cheese', 'cuisine': 'American', 'ingredients': ['Pasta', 'Cheese (Cheddar)', 'Butter', 'Milk', 'Flour']},
    {'name': 'Clam Chowder', 'cuisine': 'American', 'ingredients': ['Clam', 'Potato', 'Onion', 'Celery', 'Cream', 'Bacon']},
    {'name': 'Chicken Pot Pie', 'cuisine': 'American', 'ingredients': ['Chicken', 'Carrot', 'Peas', 'Onion', 'Cream', 'Pie Crust']},
    # Chinese
    {'name': 'Kung Pao Chicken', 'cuisine': 'Chinese', 'ingredients': ['Chicken', 'Peanut', 'Chili', 'Soy Sauce', 'Vinegar (Rice)', 'Scallion', 'Ginger']},
    {'name': 'Sweet and Sour Pork', 'cuisine': 'Chinese', 'ingredients': ['Pork', 'Pineapple', 'Bell Pepper', 'Vinegar (Rice)', 'Ketchup', 'Sugar']},
    {'name': 'Mapo Tofu', 'cuisine': 'Chinese', 'ingredients': ['Tofu', 'Pork', 'Sichuan Peppercorn', 'Chili Bean Paste', 'Soy Sauce', 'Garlic']},
    {'name': 'Char Siu (Chinese BBQ Pork)', 'cuisine': 'Chinese', 'ingredients': ['Pork', 'Soy Sauce', 'Hoisin Sauce', 'Honey', 'Five Spice Powder']},
    {'name': 'Wonton Soup', 'cuisine': 'Chinese', 'ingredients': ['Pork', 'Shrimp', 'Wonton Wrappers', 'Soy Sauce', 'Sesame Oil', 'Scallion']},
    {'name': 'Hot and Sour Soup', 'cuisine': 'Chinese', 'ingredients': ['Tofu', 'Mushroom', 'Bamboo Shoots', 'Vinegar (Rice)', 'White Pepper', 'Egg']},
    {'name': 'Dan Dan Noodles', 'cuisine': 'Chinese', 'ingredients': ['Noodles', 'Pork', 'Sichuan Peppercorn', 'Chili Oil', 'Soy Sauce', 'Sesame Paste']},
    {'name': 'Peking Duck', 'cuisine': 'Chinese', 'ingredients': ['Duck', 'Hoisin Sauce', 'Scallion', 'Cucumber', 'Pancake']},
    # Thai
    {'name': 'Green Curry Chicken', 'cuisine': 'Thai', 'ingredients': ['Chicken', 'Coconut Milk', 'Green Curry Paste', 'Fish Sauce', 'Basil', 'Lime']},
    {'name': 'Pad Thai', 'cuisine': 'Thai', 'ingredients': ['Rice Noodles', 'Shrimp', 'Tofu', 'Fish Sauce', 'Tamarind', 'Lime', 'Peanut']},
    {'name': 'Tom Yum Soup', 'cuisine': 'Thai', 'ingredients': ['Shrimp', 'Mushroom', 'Lemongrass', 'Lime', 'Chili', 'Fish Sauce']},
    {'name': 'Massaman Curry', 'cuisine': 'Thai', 'ingredients': ['Beef', 'Coconut Milk', 'Massaman Curry Paste', 'Potato', 'Onion', 'Peanut']},
    {'name': 'Som Tum (Green Papaya Salad)', 'cuisine': 'Thai', 'ingredients': ['Green Papaya', 'Fish Sauce', 'Lime', 'Chili', 'Garlic', 'Peanut']},
    {'name': 'Panang Curry', 'cuisine': 'Thai', 'ingredients': ['Beef', 'Coconut Milk', 'Panang Curry Paste', 'Fish Sauce', 'Lime']},
    # Vietnamese
    {'name': 'Pho Bo (Beef Noodle Soup)', 'cuisine': 'Vietnamese', 'ingredients': ['Beef', 'Rice Noodles', 'Onion', 'Ginger', 'Star Anise', 'Cinnamon', 'Basil', 'Lime']},
    {'name': 'Bun Cha (Grilled Pork with Noodles)', 'cuisine': 'Vietnamese', 'ingredients': ['Pork', 'Rice Noodles', 'Fish Sauce', 'Sugar', 'Garlic', 'Lettuce', 'Mint']},
    {'name': 'Banh Mi', 'cuisine': 'Vietnamese', 'ingredients': ['Pork', 'Baguette', 'Cilantro', 'Jalapeno', 'Carrot', 'Radish', 'Mayonnaise']},
    # Greek
    {'name': 'Greek Salad', 'cuisine': 'Greek', 'ingredients': ['Tomato', 'Cucumber', 'Onion', 'Feta Cheese', 'Olive', 'Oregano']},
    {'name': 'Chicken Souvlaki', 'cuisine': 'Greek', 'ingredients': ['Chicken', 'Lemon', 'Garlic', 'Oregano', 'Olive Oil', 'Yogurt']},
    {'name': 'Moussaka', 'cuisine': 'Greek', 'ingredients': ['Eggplant', 'Lamb', 'Tomato', 'Onion', 'Cinnamon', 'Bechamel Sauce']},
    # Spanish
    {'name': 'Paella Valenciana', 'cuisine': 'Spanish', 'ingredients': ['Rice', 'Chicken', 'Rabbit', 'Saffron', 'Rosemary', 'Green Beans', 'Tomato']},
    {'name': 'Gambas al Ajillo (Garlic Shrimp)', 'cuisine': 'Spanish', 'ingredients': ['Shrimp', 'Garlic', 'Olive Oil', 'Chili', 'Parsley']},
    {'name': 'Patatas Bravas', 'cuisine': 'Spanish', 'ingredients': ['Potato', 'Tomato', 'Paprika', 'Garlic', 'Olive Oil']},
    {'name': 'Tortilla Española', 'cuisine': 'Spanish', 'ingredients': ['Potato', 'Onion', 'Egg', 'Olive Oil']},
    # French
    {'name': 'Coq au Vin', 'cuisine': 'French', 'ingredients': ['Chicken', 'Red Wine', 'Mushroom', 'Onion', 'Bacon', 'Thyme']},
    {'name': 'Beef Bourguignon', 'cuisine': 'French', 'ingredients': ['Beef', 'Red Wine', 'Carrot', 'Onion', 'Mushroom', 'Garlic', 'Thyme']},
    {'name': 'French Onion Soup', 'cuisine': 'French', 'ingredients': ['Onion', 'Beef Broth', 'Thyme', 'Bay Leaf', 'Baguette', 'Gruyere Cheese']},
    {'name': 'Ratatouille', 'cuisine': 'French', 'ingredients': ['Eggplant', 'Zucchini', 'Tomato', 'Bell Pepper', 'Onion', 'Garlic', 'Basil']},
    # Middle Eastern
    {'name': 'Hummus', 'cuisine': 'Middle Eastern', 'ingredients': ['Chickpeas', 'Tahini', 'Lemon', 'Garlic', 'Olive Oil', 'Cumin']},
    {'name': 'Chicken Shawarma', 'cuisine': 'Middle Eastern', 'ingredients': ['Chicken', 'Yogurt', 'Garlic', 'Lemon', 'Cumin', 'Coriander', 'Turmeric']},
    # Turkish
    {'name': 'Doner Kebab', 'cuisine': 'Turkish', 'ingredients': ['Lamb', 'Yogurt', 'Garlic', 'Oregano', 'Paprika', 'Pita Bread']},
    {'name': 'Lentil Soup (Mercimek Çorbası)', 'cuisine': 'Turkish', 'ingredients': ['Lentils', 'Onion', 'Carrot', 'Tomato Paste', 'Mint', 'Paprika']},
    # Lebanese
    {'name': 'Tabbouleh', 'cuisine': 'Lebanese', 'ingredients': ['Parsley', 'Mint', 'Bulgur', 'Tomato', 'Onion', 'Lemon', 'Olive Oil']},
    {'name': 'Fattoush Salad', 'cuisine': 'Lebanese', 'ingredients': ['Lettuce', 'Cucumber', 'Tomato', 'Radish', 'Pita Bread', 'Sumac', 'Lemon']},
    # Ethiopian
    {'name': 'Doro Wat (Chicken Stew)', 'cuisine': 'Ethiopian', 'ingredients': ['Chicken', 'Onion', 'Berbere Spice', 'Garlic', 'Ginger', 'Butter']},
    {'name': 'Misir Wot (Red Lentil Stew)', 'cuisine': 'Ethiopian', 'ingredients': ['Lentils', 'Onion', 'Berbere Spice', 'Garlic', 'Tomato']},
    # British
    {'name': 'Fish and Chips', 'cuisine': 'British', 'ingredients': ['Fish', 'Potato', 'Flour', 'Beer', 'Peas']},
    {'name': 'Shepherd\'s Pie', 'cuisine': 'British', 'ingredients': ['Lamb', 'Potato', 'Onion', 'Carrot', 'Worcestershire Sauce']},
    # German
    {'name': 'Schnitzel', 'cuisine': 'German', 'ingredients': ['Pork', 'Flour', 'Egg', 'Breadcrumbs', 'Lemon']},
    {'name': 'Sauerbraten', 'cuisine': 'German', 'ingredients': ['Beef', 'Vinegar', 'Onion', 'Carrot', 'Ginger', 'Clove']},
    # Russian
    {'name': 'Borscht', 'cuisine': 'Russian', 'ingredients': ['Beetroot', 'Cabbage', 'Potato', 'Carrot', 'Onion', 'Beef', 'Dill']},
    {'name': 'Beef Stroganoff', 'cuisine': 'Russian', 'ingredients': ['Beef', 'Mushroom', 'Onion', 'Sour Cream', 'Mustard']},
    # Brazilian
    {'name': 'Feijoada', 'cuisine': 'Brazilian', 'ingredients': ['Black Beans', 'Pork', 'Sausage', 'Onion', 'Garlic', 'Orange']},
    # Peruvian
    {'name': 'Lomo Saltado', 'cuisine': 'Peruvian', 'ingredients': ['Beef', 'Onion', 'Tomato', 'Soy Sauce', 'Vinegar', 'Potato', 'Cilantro']},
    # Moroccan
    {'name': 'Chicken Tagine', 'cuisine': 'Moroccan', 'ingredients': ['Chicken', 'Onion', 'Ginger', 'Turmeric', 'Cinnamon', 'Olive', 'Lemon']},
    {'name': 'Couscous with Seven Vegetables', 'cuisine': 'Moroccan', 'ingredients': ['Couscous', 'Carrot', 'Zucchini', 'Turnip', 'Chickpeas', 'Cinnamon', 'Ginger']},
]


def recommend_recipes(available_ingredients, recipes, top_n=3, substitution_threshold=0.5):
    """
    Recommends recipes based on available ingredients, using the flavor model for substitutions.
    """
    scored_recipes = []
    available_set = set(available_ingredients)
    
    for recipe in recipes:
        total_score = 0
        match_details = []

        for required_ingredient in recipe['ingredients']:
            if required_ingredient in available_set:
                total_score += 1.0
                match_details.append(f"Have '{required_ingredient}'")
            else:
                best_substitute_score = 0
                best_substitute = None
                for available_ing in available_ingredients:
                    similarity = calculate_similarity(required_ingredient, available_ing)
                    if isinstance(similarity, (int, float)) and similarity > best_substitute_score:
                        best_substitute_score = similarity
                        best_substitute = available_ing
                
                if best_substitute and best_substitute_score >= substitution_threshold:
                    total_score += best_substitute_score
                    match_details.append(f"Substitute '{required_ingredient}' with '{best_substitute}' (Score: {best_substitute_score:.2f})")
                else:
                    match_details.append(f"Missing '{required_ingredient}' (No good substitute)")

        normalized_score = total_score / len(recipe['ingredients'])
        
        scored_recipes.append({
            'name': recipe['name'],
            'cuisine': recipe['cuisine'],
            'score': normalized_score,
            'details': match_details
        })

    return sorted(scored_recipes, key=lambda x: x['score'], reverse=True)[:top_n]


def get_recipe_vector(recipe_ingredients, model):
    """
    Calculates the average vector for a list of ingredients, representing a recipe's flavor profile.
    """
    vectors = []
    for ingredient in recipe_ingredients:
        try:
            vectors.append(model.wv[ingredient])
        except KeyError:
            # Ingredient not in vocabulary, skip it
            pass
    
    if not vectors:
        return None
    
    # Return the mean of all ingredient vectors
    return np.mean(vectors, axis=0)


def recommend_based_on_favorites(favorite_recipe_names, all_recipes, model, top_n=3):
    """
    Recommends new recipes based on the flavor profile of a user's favorite recipes.
    """
    favorite_ingredients = []
    favorite_recipe_set = set(favorite_recipe_names)
    for recipe in all_recipes:
        if recipe['name'] in favorite_recipe_set:
            favorite_ingredients.extend(recipe['ingredients'])
    
    if not favorite_ingredients:
        print("Could not find any of the favorite recipes in the database.")
        return []

    # Create the user's "favorite flavor" profile vector
    user_profile_vector = get_recipe_vector(favorite_ingredients, model)
    if user_profile_vector is None:
        print("Could not create a flavor profile from the favorite recipes.")
        return []

    scored_recipes = []
    for recipe in all_recipes:
        # Don't recommend a recipe the user already likes
        if recipe['name'] not in favorite_recipe_set:
            recipe_vector = get_recipe_vector(recipe['ingredients'], model)
            if recipe_vector is not None:
                # Calculate cosine similarity between user's profile and the recipe's profile
                # Using dot product and norms for cosine similarity calculation
                dot_product = np.dot(user_profile_vector, recipe_vector)
                norm_user = np.linalg.norm(user_profile_vector)
                norm_recipe = np.linalg.norm(recipe_vector)
                
                if norm_user > 0 and norm_recipe > 0:
                    similarity = dot_product / (norm_user * norm_recipe)
                    scored_recipes.append({
                        'name': recipe['name'],
                        'cuisine': recipe['cuisine'],
                        'score': similarity
                    })

    return sorted(scored_recipes, key=lambda x: x['score'], reverse=True)[:top_n]


if __name__ == "__main__":
    # --- Example 1: Based on available ingredients ---
    my_ingredients = ['Chicken', 'Soy Sauce', 'Garlic', 'Onion', 'Ginger', 'Rice', 'Black Pepper', 'Thyme']
    
    print("--- 1. Recipe Recommendations based on AVAILABLE INGREDIENTS ---")
    print(f"\nGiven your ingredients: {', '.join(my_ingredients)}")
    print("-" * 30)

    recommendations = recommend_recipes(my_ingredients, RECIPES, top_n=3)

    print("\nHere are your top 3 recipe recommendations:\n")
    for rec in recommendations:
        print(f"Name: {rec['name']} ({rec['cuisine']})")
        print(f"Match Score: {rec['score']:.2f}")
        print("Details:")
        for detail in rec['details']:
            print(f"  - {detail}")
        print("\n" + "-"*20 + "\n")

    # --- Example 2: Based on favorite recipes ---
    my_favorite_dishes = ['Beef Bulgogi', 'Teriyaki Chicken', 'Char Siu (Chinese BBQ Pork)']
    
    print("\n--- 2. Recipe Recommendations based on YOUR FAVORITE DISHES ---")
    print(f"\nBecause you like: {', '.join(my_favorite_dishes)}")
    print("-" * 30)
    
    favorite_recs = recommend_based_on_favorites(my_favorite_dishes, RECIPES, model, top_n=3)

    print("\nHere are 3 new recipes you might enjoy:\n")
    for rec in favorite_recs:
        print(f"Name: {rec['name']} ({rec['cuisine']})")
        print(f"Flavor Similarity Score: {rec['score']:.2f}")
    print("\n" + "-"*20 + "\n")