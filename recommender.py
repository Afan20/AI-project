import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import random

class FoodRecommender(nn.Module):
    def __init__(self, input_size):
        super(FoodRecommender, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

def preprocess_data(df):
    from sklearn.preprocessing import StandardScaler
    features = ['proteins_100g', 'carbohydrates_100g', 'fat_100g', 'energy-kcal_100g']
    df = df.dropna(subset=features)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler


def fetch_food_data(search_term, page_size=20):
    import requests
    import pandas as pd

    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
    'search_terms': search_term,
    'search_simple': 1,
    'action': 'process',
    'json': 1,
    'page_size': page_size,
    'fields': 'product_name,nutriments',
    'lang': 'en',
    'page': random.randint(1, 5) 
}

    response = requests.get(url, params=params)
    products = response.json().get('products', [])
    
    items = []
    for product in products:
        name = product.get('product_name', '')
        if not name or not name.isascii():
         continue
        nutriments = product.get('nutriments', {})
        items.append({
            'product_name': product.get('product_name', 'N/A'),
            'proteins_100g': nutriments.get('proteins_100g', 0),
            'carbohydrates_100g': nutriments.get('carbohydrates_100g', 0),
            'fat_100g': nutriments.get('fat_100g', 0),
            'energy-kcal_100g': nutriments.get('energy-kcal_100g', 0)
        })
    return pd.DataFrame(items)


def main():
    preference = input("Choose your preference (high protein / vegetarian / vegan): ").strip().lower()

    # Define what to search for
    if preference == "high protein":
        terms = ["chicken", "tofu", "lentils"]
    elif preference == "vegetarian":
        terms = ["vegetarian", "vegetable soup", "tofu", "paneer"]
    elif preference == "vegan":
        terms = ["vegan", "plant-based", "soy", "tempeh"]
    else:
        print("Unknown preference, using 'chicken' as default.")
        terms = ["chicken"]

    # Fetch and combine all results
    all_df = pd.DataFrame()
    for term in terms:
        df = fetch_food_data(term)
        all_df = pd.concat([all_df, df], ignore_index=True)

    # Preprocess
    if all_df.empty:
        print("No data found.")
        return

    df_processed, scaler = preprocess_data(all_df)
    features = ['proteins_100g', 'carbohydrates_100g', 'fat_100g', 'energy-kcal_100g']
    X = torch.tensor(df_processed[features].values, dtype=torch.float32)

    user_pref = torch.tensor([1.0, -0.5, -0.5, -0.2])
    y = torch.matmul(X, user_pref).unsqueeze(1)

    model = FoodRecommender(input_size=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = model(X).squeeze().numpy()

    df_processed['score'] = scores
    recommendations = df_processed.sort_values(by='score', ascending=False)

    print("\nTop Recommendations:\n")
    print(recommendations[['product_name', 'score']].head(10))

def get_recommendations(preference):
    preference = preference.lower().strip()

    # 🔍 Keyword mapping
    keyword_map = {
        "high protein": ["chicken", "tofu", "lentils"],
        "more dairy": ["milk", "cheese", "yogurt"],
        "low fat": ["vegetables", "fruit", "lean meat"],
        "vegan": ["plant-based", "tempeh", "soy"],
        "vegetarian": ["vegetarian", "paneer", "tofu"],
        "gluten free": ["gluten free", "rice", "corn"],
        "low sugar": ["unsweetened", "no sugar", "sugar free"]
    }

    terms = []
    for key, values in keyword_map.items():
        if key in preference:
            terms.extend(values)

    # If no match, default to some fallback
    if not terms:
        terms = ["healthy"]

    # (Then continue with fetch_food_data, preprocess, model, etc.)

    all_df = pd.DataFrame()
    for term in terms:
        df = fetch_food_data(term)
        all_df = pd.concat([all_df, df], ignore_index=True)

    if all_df.empty:
        return []

    df_processed, _ = preprocess_data(all_df)
    features = ['proteins_100g', 'carbohydrates_100g', 'fat_100g', 'energy-kcal_100g']
    X = torch.tensor(df_processed[features].values, dtype=torch.float32)

    user_pref = torch.tensor([1.0, -0.5, -0.5, -0.2])
    y = torch.matmul(X, user_pref).unsqueeze(1)

    model = FoodRecommender(input_size=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = model(X).squeeze().numpy()

    df_processed['score'] = scores
    top_results = df_processed.sort_values(by='score', ascending=False)
    return top_results[['product_name', 'score']].head(10).to_dict(orient='records')
