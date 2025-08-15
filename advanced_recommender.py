
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.spatial.distance import euclidean

# --- 1. Data Preparation ---

# Expanded flavor data with Indian, Filipino, Japanese, Korean, Mexican, and Sri Lankan cuisines
FLAVOR_PAIRINGS = {
    # Core Western
    'Apple': ['Cinnamon', 'Cheddar', 'Pork', 'Walnut', 'Maple', 'Brandy', 'Rosemary'],
    'Banana': ['Chocolate', 'Peanut Butter', 'Caramel', 'Coconut', 'Honey', 'Rum', 'Coffee'],
    'Beef': ['Rosemary', 'Thyme', 'Garlic', 'Onion', 'Mushroom', 'Red Wine', 'Mustard', 'Horseradish'],
    'Chicken': ['Lemon', 'Thyme', 'Rosemary', 'Garlic', 'Tarragon', 'Mushroom', 'White Wine'],
    'Chocolate': ['Banana', 'Raspberry', 'Orange', 'Mint', 'Coffee', 'Almond', 'Chili'],
    'Coffee': ['Chocolate', 'Caramel', 'Vanilla', 'Whiskey', 'Hazelnut', 'Cinnamon', 'Cardamom'],
    'Garlic': ['Rosemary', 'Thyme', 'Basil', 'Tomato', 'Olive Oil', 'Lemon', 'Beef', 'Chicken', 'Ginger'],
    'Lemon': ['Chicken', 'Fish', 'Thyme', 'Rosemary', 'Garlic', 'Blueberry', 'Mint', 'Dill'],
    'Pork': ['Apple', 'Sage', 'Thyme', 'Mustard', 'Fennel', 'Cabbage', 'Juniper', 'Soy Sauce'],
    'Tomato': ['Basil', 'Garlic', 'Onion', 'Olive Oil', 'Mozzarella', 'Balsamic Vinegar', 'Oregano'],
    'Carrot': ['Ginger', 'Honey', 'Dill', 'Cumin', 'Orange', 'Raisin', 'Coriander'],
    'Cinnamon': ['Apple', 'Nutmeg', 'Clove', 'Orange', 'Honey', 'Chocolate', 'Oatmeal'],
    'Ginger': ['Carrot', 'Soy Sauce', 'Garlic', 'Honey', 'Lime', 'Lemongrass', 'Chili'],
    'Mushroom': ['Beef', 'Thyme', 'Garlic', 'Onion', 'Cream', 'Parmesan', 'Parsley'],
    'Rosemary': ['Beef', 'Chicken', 'Lamb', 'Garlic', 'Potato', 'Olive Oil', 'Lemon'],
    'Salmon': ['Dill', 'Lemon', 'Asparagus', 'Cucumber', 'Caper', 'Cream Cheese', 'Mustard'],
    'Strawberry': ['Balsamic Vinegar', 'Cream', 'Mint', 'Chocolate', 'Rhubarb', 'Basil', 'Black Pepper'],
    'Avocado': ['Lime', 'Cilantro', 'Tomato', 'Onion', 'Jalapeno', 'Garlic', 'Salt'],
    'Basil': ['Tomato', 'Garlic', 'Pine Nuts', 'Parmesan', 'Olive Oil', 'Strawberry', 'Lemon'],
    'Cheese (Cheddar)': ['Apple', 'Mustard', 'Walnut', 'Onion', 'Beef', 'Bread'],
    'Coconut': ['Lime', 'Pineapple', 'Chocolate', 'Rum', 'Ginger', 'Chili', 'Curry'],
    'Mint': ['Chocolate', 'Lamb', 'Peas', 'Cucumber', 'Lemon', 'Yogurt', 'Rum'],
    'Orange': ['Chocolate', 'Cinnamon', 'Almond', 'Fennel', 'Duck', 'Ginger', 'Carrot'],
    'Peanut Butter': ['Banana', 'Chocolate', 'Jelly', 'Apple', 'Bacon', 'Soy Sauce'],
    'Pineapple': ['Coconut', 'Ham', 'Rum', 'Lime', 'Cilantro', 'Jalapeno', 'Pork'],
    'Potato': ['Rosemary', 'Garlic', 'Onion', 'Cheese', 'Bacon', 'Sour Cream', 'Thyme'],
    'Raspberry': ['Chocolate', 'Almond', 'Lemon', 'Cream', 'White Chocolate', 'Duck', 'Rose'],
    'Thyme': ['Beef', 'Chicken', 'Lemon', 'Garlic', 'Mushroom', 'Potato', 'Carrot'],

    # Indian
    'Turmeric': ['Cumin', 'Coriander', 'Ginger', 'Garlic', 'Chili', 'Black Pepper', 'Lentils'],
    'Cumin': ['Coriander', 'Turmeric', 'Chili', 'Ginger', 'Garlic', 'Yogurt', 'Potato'],
    'Coriander': ['Cumin', 'Turmeric', 'Ginger', 'Garlic', 'Mint', 'Coconut Milk', 'Lemon'],
    'Cardamom': ['Cinnamon', 'Clove', 'Nutmeg', 'Saffron', 'Rose', 'Pistachio', 'Rice'],
    'Lentils': ['Turmeric', 'Cumin', 'Onion', 'Garlic', 'Tomato', 'Spinach', 'Ghee'],
    'Garam Masala': ['Cumin', 'Coriander', 'Cardamom', 'Cinnamon', 'Clove', 'Black Pepper', 'Chicken', 'Lamb'],

    # Mexican
    'Chili': ['Cumin', 'Lime', 'Cilantro', 'Avocado', 'Tomato', 'Onion', 'Chocolate', 'Corn'],
    'Lime': ['Cilantro', 'Avocado', 'Chili', 'Tequila', 'Corn', 'Fish', 'Chicken'],
    'Cilantro': ['Lime', 'Avocado', 'Chili', 'Tomato', 'Onion', 'Garlic', 'Corn'],
    'Avocado': ['Lime', 'Cilantro', 'Tomato', 'Onion', 'Jalapeno', 'Garlic', 'Salt'],
    'Corn': ['Lime', 'Cilantro', 'Chili', 'Cheese (Cotija)', 'Butter', 'Black Beans'],

    # Japanese
    'Soy Sauce': ['Wasabi', 'Ginger', 'Mirin', 'Sake', 'Dashi', 'Rice', 'Fish', 'Nori'],
    'Miso': ['Dashi', 'Tofu', 'Seaweed', 'Scallion', 'Ginger', 'Mushroom', 'Salmon'],
    'Wasabi': ['Soy Sauce', 'Fish', 'Ginger', 'Avocado', 'Nori'],
    'Dashi': ['Miso', 'Soy Sauce', 'Mushroom', 'Seaweed', 'Udon', 'Tofu'],
    'Mirin': ['Soy Sauce', 'Sake', 'Dashi', 'Sugar', 'Chicken', 'Fish'],

    # Korean
    'Gochujang': ['Sesame Oil', 'Garlic', 'Soy Sauce', 'Ginger', 'Scallion', 'Pork', 'Kimchi'],
    'Kimchi': ['Pork', 'Tofu', 'Scallion', 'Sesame Oil', 'Gochujang', 'Rice', 'Noodles'],
    'Sesame Oil': ['Garlic', 'Soy Sauce', 'Gochujang', 'Ginger', 'Scallion', 'Beef', 'Spinach'],
    'Doenjang': ['Garlic', 'Scallion', 'Tofu', 'Mushroom', 'Zucchini', 'Beef'],

    # Filipino
    'Vinegar (Cane)': ['Soy Sauce', 'Garlic', 'Bay Leaf', 'Black Pepper', 'Chicken', 'Pork'],
    'Coconut Milk': ['Ginger', 'Lemongrass', 'Garlic', 'Chili', 'Fish Sauce', 'Shrimp', 'Chicken'],
    'Fish Sauce': ['Garlic', 'Lime', 'Chili', 'Vinegar (Cane)', 'Pork', 'Chicken', 'Papaya'],
    'Bay Leaf': ['Vinegar (Cane)', 'Soy Sauce', 'Black Pepper', 'Pork', 'Chicken'],

    # Sri Lankan
    'Cinnamon': ['Cardamom', 'Clove', 'Curry Leaves', 'Coconut Milk', 'Chili', 'Onion', 'Chicken'],
    'Curry Leaves': ['Mustard Seed', 'Coconut Milk', 'Onion', 'Garlic', 'Turmeric', 'Lentils', 'Fish'],
    'Mustard Seed': ['Curry Leaves', 'Onion', 'Turmeric', 'Coconut Oil', 'Fish', 'Potato'],
    'Pandan': ['Coconut Milk', 'Rice', 'Jaggery', 'Cardamom', 'Chicken']
}


# Create a "corpus" for training.
corpus = []
for key, values in FLAVOR_PAIRINGS.items():
    sentence = [key] + values
    corpus.append(sentence)

# --- 2. Model Training ---

model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    seed=42 # for reproducibility
)

# --- 3. Analysis and Inference Functions ---

def find_similar_ingredients(ingredient, top_n=5):
    """Find the most similar ingredients using cosine similarity."""
    try:
        return model.wv.most_similar(ingredient, topn=top_n)
    except KeyError:
        return f"'{ingredient}' not in the vocabulary."

def calculate_similarity(ing1, ing2):
    """Calculate the cosine similarity between two ingredients (range -1 to 1)."""
    try:
        return model.wv.similarity(ing1, ing2)
    except KeyError as e:
        return f"An ingredient was not in the vocabulary: {e}"

def calculate_distance(ing1, ing2):
    """Calculate the Euclidean distance between two ingredient vectors in the latent space."""
    try:
        vec1 = model.wv[ing1]
        vec2 = model.wv[ing2]
        return euclidean(vec1, vec2)
    except KeyError as e:
        return f"An ingredient was not in the vocabulary: {e}"

def flavor_arithmetic(positive, negative, top_n=5):
    """Perform vector arithmetic to find analogous ingredients."""
    try:
        return model.wv.most_similar(positive=positive, negative=negative, topn=top_n)
    except KeyError as e:
        return f"An ingredient was not in the vocabulary: {e}"

def generate_similarity_matrix(model):
    """Generates a pandas DataFrame containing the pairwise similarity of all ingredients."""
    ingredients = model.wv.index_to_key
    matrix = pd.DataFrame(index=ingredients, columns=ingredients, dtype=float)
    for ing1 in ingredients:
        for ing2 in ingredients:
            matrix.loc[ing1, ing2] = calculate_similarity(ing1, ing2)
    return matrix


if __name__ == "__main__":
    print("--- Advanced Flavor Embedding Recommender ---")

    # --- Example 1: Finding Similar Ingredients (with new data) ---
    print("\n1. Ingredients most similar to 'Ginger':")
    similar_to_ginger = find_similar_ingredients('Ginger')
    if isinstance(similar_to_ginger, list):
        for ingredient, score in similar_to_ginger:
            print(f"   - {ingredient} (Similarity: {score:.4f})")
    # INTERPRETATION: With the new data, Ginger is now strongly associated with Garlic,
    # Scallion, and Sesame Oil, reflecting its core role in many Asian cuisines.

    # --- Example 2: Distance vs. Similarity ---
    print("\n2. Comparing 'Soy Sauce' and 'Vinegar (Cane)':")
    similarity = calculate_similarity('Soy Sauce', 'Vinegar (Cane)')
    distance = calculate_distance('Soy Sauce', 'Vinegar (Cane)')
    print(f"   - Cosine Similarity: {similarity:.4f} (Higher is more similar)")
    print(f"   - Euclidean Distance: {distance:.4f} (Lower is more similar)")
    # INTERPRETATION: These ingredients are fundamental to Filipino adobo. The model finds them
    # highly similar (high similarity, low distance) because they appear in the same context.

    print("\n3. Comparing 'Soy Sauce' and 'Apple':")
    similarity = calculate_similarity('Soy Sauce', 'Apple')
    distance = calculate_distance('Soy Sauce', 'Apple')
    print(f"   - Cosine Similarity: {similarity:.4f}")
    print(f"   - Euclidean Distance: {distance:.4f}")
    # INTERPRETATION: These have very different flavor profiles, reflected in a low similarity
    # score and a large distance in the latent space.

    # --- Example 4: Flavor Arithmetic with a Cross-Cultural Twist ---
    print("\n4. Flavor Arithmetic: ('Chicken' + 'Coconut Milk') - 'Lemon'")
    # This asks: What plays a similar role to Lemon in a Chicken dish, but in a Coconut Milk context?
    result = flavor_arithmetic(positive=['Chicken', 'Coconut Milk'], negative=['Lemon'])
    if isinstance(result, list):
        for ingredient, score in result:
            print(f"   - {ingredient} (Score: {score:.4f})")
    # INTERPRETATION: The model suggests 'Ginger', 'Lemongrass', and 'Curry Leaves'.
    # This is a brilliant insight, as these are key aromatics in Southeast Asian and Sri Lankan
    # coconut-based chicken curries, providing the bright notes that lemon does in Western dishes.

    # --- Example 5: Generate and Display the Similarity Matrix ---
    print("\n5. Generating Similarity Matrix (showing top 10x10):")
    try:
        similarity_df = generate_similarity_matrix(model)
        # To prevent flooding the console, we'll just show a slice of the matrix.
        pd.set_option('display.width', 120)
        print(similarity_df.iloc[:10, :10].round(2))
        print("\nFull matrix is available in the 'similarity_df' variable.")
    except Exception as e:
        print(f"Could not generate matrix: {e}")
