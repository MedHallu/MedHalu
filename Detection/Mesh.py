import xml.etree.ElementTree as ET
import pandas as pd
import ast
from tqdm import tqdm 
import re
from collections import Counter
import matplotlib.pyplot as plt


from tqdm import tqdm
pd.set_option('display.max_colwidth', None)

# Path to your downloaded MeSH XML file
mesh_xml_file = './desc2025.xml'

# Parse the XML file
tree = ET.parse(mesh_xml_file)
root = tree.getroot()

# Dictionary mapping broad category letters to their names
broad_categories = {
    'A': 'Anatomy',
    'B': 'Organisms',
    'C': 'Diseases',
    'D': 'Chemicals and Drugs',
    'E': 'Analytical, Diagnostic and Therapeutic Techniques and Equipment',
    'F': 'Psychiatry and Psychology',
    'G': 'Phenomena and Processes',
    'H': 'Disciplines and Occupations',
    'I': 'Anthropology, Education, Sociology, and Social Phenomena',
    'J': 'Technology, Industry, and Agriculture',
    'K': 'Humanities',
    'L': 'Information Science',
    'M': 'Persons',
    'N': 'Health Care',
    'V': 'Publication Characteristics',
    'Z': 'Geographicals'
}

# Function to get the name of the highest level (broad category) for a MeSH term
from collections import Counter

def get_broad_category(term):
    for descriptor in root.findall(".//DescriptorRecord"):
        name = descriptor.find("DescriptorName/String").text
        if name == term:
            # Find all tree numbers associated with this MeSH term
            tree_numbers = descriptor.findall(".//TreeNumber")
            
            # If no tree numbers are found, return "Unknown Category"
            if not tree_numbers:
                print(f"No tree numbers found for {term}")
                return "Unknown Category"
            
            print(f"Tree numbers for {term}:")
            for tree in tree_numbers:
                print(tree.text)
            
            # Extract the first letters of the tree numbers (the broad categories)
            broad_category_letters = [tree.text[0] for tree in tree_numbers if tree.text]
            
            # Count the occurrences of each broad category letter
            category_counts = Counter(broad_category_letters)
            
            # Find the most appeared broad category letter
            most_common_letter, _ = category_counts.most_common(1)[0]
            
            # Get the name of the broad category using the dictionary
            broad_category_name = broad_categories.get(most_common_letter, "Unknown Category")
            
            return broad_category_name
    return "Unknown Category"

# Example usage
term = "Risk Assessment"
highest_level = get_broad_category(term)
print(f"Highest level for {term}: {highest_level}")


# pubmedqa labled

pqa_labeled= pd.read_csv("./pqa_artificial.csv")
pqa_labeled.head()

pqa_labeled = pqa_labeled[:9000]

# Define a function to clean the 'context' strings and extract 'meshes'
def clean_and_extract_meshes(raw_str):
    # Step 1: Remove newlines, dtype declarations, and extra annotations
    cleaned_str = raw_str.replace("\n", " ").replace("array(", "").replace("dtype=object", "").replace(")", "")
    
    # Step 2: Trim excess whitespace
    cleaned_str = ' '.join(cleaned_str.split())
    
    # Step 3: Use regex to extract the 'meshes' array from the cleaned string
    pattern = r"'meshes':\s*\[(.*?)\]"
    matches = re.search(pattern, cleaned_str)
    
    # Step 4: If matches found, return the extracted 'meshes', otherwise return None
    if matches:
        # Split by commas but account for multi-word terms that should stay together
        meshes_list = [item.strip().strip("'") for item in matches.group(1).split("', '")]
        return '; '.join(meshes_list)  # Use semicolon and space as a separator to avoid conflicts with commas
    else:
        return None

# # Apply the cleaning and extraction function to the 'context' column
# pqa_labeled['meshes'] = pqa_labeled['context'].apply(clean_and_extract_meshes)

# Define a function to remove duplicates and maintain multi-word terms with commas
def remove_duplicates(meshes_str):
    if meshes_str:
        # Split the string by semicolons to get individual items
        meshes_list = [mesh.strip() for mesh in meshes_str.split(";")]
        # Remove duplicates by converting the list to a set, then sort and join back
        unique_meshes = '; '.join(sorted(set(meshes_list)))
        return unique_meshes
    return meshes_str

# Apply the function to the 'meshes' column
pqa_labeled['meshes'] = pqa_labeled['meshes'].apply(remove_duplicates)


pqa_labeled.head()

# Enable the use of tqdm with pandas' apply method
tqdm.pandas()

# Function to process each row in the 'meshes' column with progress tracking
def process_meshes(row_meshes):
    categories = []
    if row_meshes:
        # Split the meshes by semicolons
        terms = [term.strip() for term in row_meshes.split(";")]
        
        # Apply the get_broad_category function to each mesh term
        for term in terms:
            category = get_broad_category(term)
            categories.append(category)
            print(f"Term: {term} -> Category: {category}")  # Debug log
            print("-------------------------------------")
    return categories

# # Apply the function to each row and create a new column 'categories' with a progress bar
# pqa_labeled['categories'] = pqa_labeled['meshes'].progress_apply(process_meshes)

from collections import Counter

# Using inverse term frequency to assign weights for each mesh terms
# Step 1: Calculate term frequencies across all rows
# Flatten all MeSH terms and count frequencies
all_mesh_terms = [mesh.strip() for meshes in pqa_labeled['meshes'] for mesh in meshes.split('; ')]
mesh_term_frequencies = Counter(all_mesh_terms)

# Step 2: Assign weights (inverse of term frequency)
mesh_weights = {term: 1 / freq for term, freq in mesh_term_frequencies.items()}

# Step 3: Build a mapping of MeSH terms to their categories from the dataset
mesh_category_mapping = {}

for _, row in pqa_labeled.iterrows():
    meshes = row['meshes'].split('; ')
    
    # Ensure categories is a string before using eval
    if isinstance(row['categories'], str):
        categories = eval(row['categories'])  # Convert string representation of list to actual list
    else:
        categories = row['categories']  # Assume it's already a list or appropriate format
    
    # Map each MeSH term to its category
    for mesh, category in zip(meshes, categories):
        mesh_category_mapping[mesh.strip()] = category.strip()

# Step 4: Function to classify each row based on the cumulative weights of categories
def classify_question_by_cumulative_weight(mesh_terms):
    category_weights = Counter()
    for mesh in mesh_terms.split('; '):
        mesh = mesh.strip()
        weight = mesh_weights.get(mesh, 0)  # Default weight is 0 if term is not in the dictionary
        category = mesh_category_mapping.get(mesh, "Unknown")
        category_weights[category] += weight  # Add weight to the category

    # Select the category with the highest cumulative weight
    if category_weights:
        return max(category_weights.items(), key=lambda x: x[1])[0]
    else:
        return "Unknown"

# # Step 5: Apply classification to each row
pqa_labeled['mesh_classification'] = pqa_labeled['meshes'].apply(classify_question_by_cumulative_weight)

pqa_labeled.to_csv("./pqa_labeled_add_category_inverse_term_frequency.csv")
pqa_labeled.head(10)


category_distribution = pqa_labeled['mesh_classification'].value_counts()
import matplotlib.pyplot as plt

# Plot the distribution of the most frequent category
plt.figure(figsize=(10, 6))
category_distribution.plot(kind='bar')

# Add titles and labels
plt.title('Distribution of Most Frequent Category', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()
category_distribution

# OLD
# Function to calculate the most frequent category for each row
def most_frequent_category(categories):
    if not categories:
        return "Unknown Category"
    
    # Use Counter to count the frequency of each category
    category_counts = Counter(categories)
    
    # Get the most common categories
    most_common_categories = category_counts.most_common()

    # Check if the most common category is "Unknown Category"
    if most_common_categories[0][0] == "Unknown Category" and len(most_common_categories) > 1:
        # Return the second most frequent category if "Unknown Category" is the most common
        return most_common_categories[1][0]
    
    # Otherwise, return the most common category
    return most_common_categories[0][0]

# Apply the function to each row and create a new column 'most_frequent_category'
pqa_labeled['most_frequent_category'] = pqa_labeled['categories'].apply(most_frequent_category)

pqa_labeled.to_csv("./pqa_labeled_add_category.csv")

category_distribution = pqa_labeled['most_frequent_category'].value_counts()


# Plot the distribution of the most frequent category
plt.figure(figsize=(10, 6))
category_distribution.plot(kind='bar')

# Add titles and labels
plt.title('Distribution of Most Frequent Category', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()
category_distribution