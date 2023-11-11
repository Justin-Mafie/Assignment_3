def get_frequency_ratio(input_string):
    # Split the input string into a list of words
    list_of_words = input_string.split(' ')

    # Initialize a dictionary to store word frequencies
    dict_of_frequencies = {}

    # Count word frequencies
    for word in list_of_words:
        if word in dict_of_frequencies:
            # Increment the count if the word is already in the dictionary
            dict_of_frequencies[word] += 1
        else:
            # Add the word to the dictionary with a count of 1 if it's not present
            dict_of_frequencies[word] = 1

    # Find the most frequent word in the text
    most_frequent_word = max(dict_of_frequencies, key=dict_of_frequencies.get)

    # Calculate the frequency ratio of the most frequent word
    frequency_ratio = dict_of_frequencies[most_frequent_word] / len(list_of_words)

    return frequency_ratio


# Example usage
input_text = "This is a sample text with repeated words. This is a sample."
result = get_frequency_ratio(input_text)
print("Frequency Ratio:", result)
