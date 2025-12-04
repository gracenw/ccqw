#include "flesch.hpp"


float FleschReadingEase::get() {
    int number_sentences = 0;
    int number_words = 0;
    int number_syllables = 0;

    string sentence = "";
    // iterate over the query to split into sentences
    for (int i = 0; i < this->query.length(); i ++) {
        // can reasonably assume there aren't going to be quotes in the query
        // but that is something i can add later too
        if (this->query[i] == '.' || this->query[i] == '?' || this->query[i] == '!') {
            number_sentences ++;
            string word = "";
            // iterate over the sentence to split into words
            for (int j = 0; j < sentence.length(); j ++) {
                if (sentence[j] == ' ' || sentence[j] == '-') {
                    // end of word, send to calculate syllable count
                    number_words ++;
                    number_syllables += numSyllables(word);
                    word = "";
                }
                else if (j == sentence.length() - 1) {
                    word += sentence[j];
                    number_words ++;
                    number_syllables += numSyllables(word);
                    word = "";
                }
                else {
                    // word continues, ignore any punctuation
                    int ascii = (unsigned char) sentence[j];
                    if ((ascii > 64 && ascii < 91) || (ascii > 96 && ascii < 123)) {
                        word += sentence[j];
                    }
                }
            }
            sentence = "";
        }
        else {
            if (!(sentence.length() == 0 && this->query[i] == ' ')) { 
                sentence += this->query[i];
            }
        }
    }
    // cout << "sentences: " << number_sentences << endl;
    // cout << "words: " << number_words << endl;
    // cout << "syllables: " << number_syllables << endl;

    // calculate flesch reading ease
    float reading_ease = 206.835 - 1.015 * (number_words / number_sentences) - 84.6 * (number_syllables / number_words);
    return reading_ease;
}

bool FleschReadingEase::isVowel(char letter) {
    if (letter == 'a' || letter == 'i' || letter == 'o' || letter == 'u' || letter == 'e') {
        return true;
    }
    else if (letter == 'A' || letter == 'I' || letter == 'O' || letter == 'U' || letter == 'E') {
        return true;
    }
    else {
        return false;
    }
}

/*
Count the number of vowels (A, E, I, O, U) in the word.
    Add 1 every time the letter 'y' makes the sound of a vowel (A, E, I, O, U).
    Subtract 1 for each silent vowel (like the silent 'e' at the end of a word).
Subtract 1 for each diphthong or triphthong in the word.
    Diphthong: when 2 vowels make only 1 sound (au, oy, oo)
    Triphthong: when 3 vowels make only 1 sound (iou)
Does the word end with "le" or "les?" Add 1 only if the letter before the "le" is a consonant.
The number you get is the number of syllables in your word.
*/

int FleschReadingEase::numSyllables(string word) {
    int number_syllables = 0;

    // assume one word - count the vowels
    for (int i = 0; i < word.length(); i ++) {
        if (isVowel(word[i])) {
            number_syllables ++;
            // check for dipththongs
            if (i < word.length() - 1) {
                if (isVowel(word[i + 1])) {
                    i ++; // TO DO: check for vowel hiatuses - eac, what else??
                    
                    // check for tripththongs
                    if (i < word.length() - 2) {
                        if (isVowel(word[i + 2])) {
                            i ++;
                        }
                    }
                }
            }
        }
    }

    // make sure there isn't a silent 'e' at end of word
    if (word[word.length() - 1] == 'e' && word.length() > 1 && !isVowel(word[word.length() - 2])) {
        number_syllables --;
    }

    // count y if no vowels
    if (number_syllables == 0) {
        for (int i= 0; i < word.length(); i ++) {
            if (word[i] == 'y') {
                number_syllables ++;
            }
        }
    }
    return number_syllables;
}