import textstat as ts
import os, time


# def flesch_reading_ease(text_input: str) -> float:
#     num_words = 
#     num_senteces = 
#     num_syllables = 
#     ease = 206.835 - 1.015


def send_to_edgebert(text_input: str):
    ## fill in with corresponding timing values
    time.sleep(1)


def send_to_calm(text_input: str):
    ## fill in with corresponding timing values
    time.sleep(1)


def send_to_cloud(text_input: str):
    ## fill in with corresponding timing values
    time.sleep(1)


def main():
    ## get textual input
    text_input = input('enter a sample query: ')

    ## parse text for grammatical/spelling errors


    ## evaluate complexity of sentence
    complexity = ts.flesch_reading_ease(text_input)

    ## send to appropriate core
    if complexity > 80:
        result = send_to_edgebert(text_input)
    else:
        encryption = send_to_calm(text_input)
        result = send_to_cloud(encryption)


if __name__ == '__main__':
    main()