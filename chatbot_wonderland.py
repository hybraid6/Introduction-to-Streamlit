
    
##### Let's build a beginner-friendly chatbot in Streamlit #####
# This project will build a chatbot that reads a text file, processes it, and returns relevant answers based on user input.

# Importing necessary libraries

# nltk (Natural Language Toolkit) library for various text processing tasks
import nltk
import streamlit as st  # Streamlit is used for building interactive web applications
from nltk.tokenize import word_tokenize, sent_tokenize  # Tokenizers for splitting text into words and sentences
from nltk.corpus import stopwords  # List of common words (stopwords) that are usually removed from text (like "is", "the", "and")
from nltk.stem import WordNetLemmatizer  # Lemmatizer to reduce words to their base form (e.g., 'running' -> 'run')
import string  # Python's built-in library for handling strings and punctuation

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load stopwords and initialize lemmatizer
stop_words = set(stopwords.words('english'))  # Load a set of common English stopwords to filter out later
lemmatizer = WordNetLemmatizer()  # Initialize a lemmatizer to reduce words to their base form

# Define a function to preprocess text (tokenizing, removing stopwords and punctuation, lemmatizing)
def preprocess(sentence):
    # Tokenize the sentence into words and convert to lowercase
    words = word_tokenize(sentence.lower())
    
    # Remove stopwords and punctuation from the list of words
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    
    # Lemmatize each word to convert it to its base form (e.g., 'running' -> 'run')
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Return the list of processed words
    return words

# Load the text file (Alice in Wonderland)
def load_text():
    try:
        # Provide the path to the text file
        file_path = 'alice_in_wonderland.txt'
        
        # Open the file, read its content, and replace newline characters with spaces
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace('\n', ' ')
    
    # Handle case where the file is not found and display an error message in Streamlit
    except FileNotFoundError:
        st.error("Text file not found.")
        return ""

# Tokenize the text into sentences and preprocess them
def prepare_corpus(text):
    # Tokenize the text into individual sentences using sent_tokenize
    sentences = sent_tokenize(text)
    
    # Preprocess each sentence (tokenizing, removing stopwords/punctuation, and lemmatizing)
    return [preprocess(sentence) for sentence in sentences]

# Calculate Jaccard similarity between two sets
def jaccard_similarity(query, sentence):
    # Convert both the query and sentence to sets (unique words)
    query_set = set(query)
    sentence_set = set(sentence)
    
    # If the union of both sets is zero, return 0 to avoid division by zero
    if len(query_set.union(sentence_set)) == 0:
        return 0
    
    # Calculate the Jaccard similarity as the size of intersection divided by the size of union
    return len(query_set.intersection(sentence_set)) / len(query_set.union(sentence_set))

# Find the most relevant sentence using Jaccard similarity
def get_most_relevant_sentence(query, corpus, original_sentences):
    # Preprocess the user query (tokenization, stopword removal, etc.)
    query = preprocess(query)
    
    # Initialize variables to store the maximum similarity and best matching sentence
    max_similarity = 0
    best_sentence = "I couldn't find a relevant answer."  # Default response if no match is found
    
    # Iterate over the corpus of preprocessed sentences to find the best match
    for i, sentence in enumerate(corpus):
        # Calculate the Jaccard similarity between the user query and the current sentence
        similarity = jaccard_similarity(query, sentence)
        
        # If the similarity score is higher than the current maximum, update the best sentence
        if similarity > max_similarity:
            max_similarity = similarity
            best_sentence = original_sentences[i]  # Retrieve the original sentence (before preprocessing)
    
    # Return the most relevant sentence (or the default response if no match is found)
    return best_sentence

# Main function to create the chatbot interface in Streamlit
def main():
    # Title for the app
    st.title("Wonderland's Novice Chatbot")
    
    # A brief description of the chatbot's purpose
    st.write("Hello! Ask me anything related to Alice in Wonderland!")
    
    # Add a dropdown (expander) for suggested questions
    with st.expander("Click me for suggestions"):
        st.write("""
        1. Who does Alice meet first in Wonderland?
        2. What is the Cheshire Cat's famous line?
        3. How does Alice enter Wonderland?
        4. What is the Queen of Hearts known for?
        5. Why did Alice follow the White Rabbit?
        6. What was Alice's reaction to the Mad Hatter's tea party?
        7. What advice does the Caterpillar give Alice?
        8. What is the significance of the bottle labeled 'Drink Me'?
        9. How does the story of Alice in Wonderland end?
        10. What game does the Queen of Hearts play with Alice?
        """)
    # Load and prepare text corpus
    text = load_text()  # Load the text from the file (Alice in Wonderland)
    if text:
        # Preprocess the text to create a corpus of tokenized sentences
        corpus = prepare_corpus(text)  # Prepares the text into a list of preprocessed sentences
        original_sentences = sent_tokenize(text)  # Tokenizes the original text into sentences for later reference

        # Get user input from the Streamlit interface
        user_input = st.text_input("Enter your question:")  # Input field for the user's question

        # If the user clicks the submit button
        if st.button("Submit"):
            if user_input:
                # Get the most relevant sentence from the corpus based on the user's input
                response = get_most_relevant_sentence(user_input, corpus, original_sentences)
                st.write(f"Chatbot: {response}")  # Display the chatbot's response
            else:
                st.write("Please enter a question.")  # Prompt user to enter a question if the input is empty

# Run the Streamlit app
if __name__ == "__main__":
    main()  # Call the main function to run the Streamlit app
    
