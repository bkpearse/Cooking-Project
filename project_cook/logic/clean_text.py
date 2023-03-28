#basic cleaning
def basic_cleaning(text):
    text = text.strip()
    text = text.lower()
    return text


#Removing punctuation
def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', '"[]'))
    text = text.replace('\\u00b0', ' degree')
    return text


def remove_words(text):
    words_to_remove = [
        'brown', 'breasts', 'frozen', 'crumbs', 'spread', 'kosher', 'ground',
        'chopped', 'diced', 'minced', 'sliced', 'grated', 'shredded',
        'boneless', 'skinless', 'organic', 'fresh', 'canned', 'low-fat',
        'low-sodium', 'gluten-free', 'halal', 'all-purpose', 'seasoning',
        'spice', 'sauce', 'paste', 'juice', 'zest', 'extract', 'powder',
        'cream', 'sour', 'soy', 'toasted', 'roasted', 'smoked', 'grilled',
        'fried', 'baked', 'boiled', 'steamed', 'stir-fried', 'sauteed',
        'white', 'sesame', 'of', 'powdered', 'shredded', 'size'
    ]

    for word in words_to_remove:
        text = text.replace(word, '')

    return text.strip()
