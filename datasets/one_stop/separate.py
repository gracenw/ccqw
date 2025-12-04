import os
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    root = '/home/gracen/repos/peace/data'
    separated_loc = root + '/OneStopEnglishCorpus/Texts-SeparatedByReadingLevel' 
    texts = []
    labels = []
    ele_dir = separated_loc + '/Ele-Txt' # 0
    int_dir = separated_loc + '/Int-Txt' # 1
    adv_dir = separated_loc + '/Adv-Txt' # 2
    for file_name in os.listdir(ele_dir):
        if file_name.endswith(".txt"):
            with open(ele_dir + '/' + file_name, 'r', encoding='utf-8-sig') as file:
                data = file.read().replace('\n', '').rstrip().lower()
                texts.append(data)
                labels.append(0)
    for file_name in os.listdir(int_dir):
        if file_name.endswith(".txt"):
            with open(int_dir + '/' + file_name, 'r', encoding='utf-8-sig') as file:
                data = file.read().replace('Intermediate', '', 1).replace('\n', '').strip().lower()
                texts.append(data)
                labels.append(0)
    for file_name in os.listdir(adv_dir):
        if file_name.endswith(".txt"):
            with open(adv_dir + '/' + file_name, 'r', encoding='utf-8-sig') as file:
                data = file.read().replace('\n', '').rstrip().lower()
                texts.append(data)
                labels.append(1)
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

    new_folder = '/home/gracen/repos/peace/data/processed'
    train_folder = new_folder + '/train'
    test_folder = new_folder + '/test'

    with open(train_folder + '/texts.txt', 'w+') as file:
        for text in train_texts:
            file.write(text + '\n')
    
    with open(train_folder + '/labels.txt', 'w+') as file:
        for label in train_labels:
            file.write(str(label) + '\n')

    with open(test_folder + '/texts.txt', 'w+') as file:
        for text in test_texts:
            file.write(text + '\n')
    
    with open(test_folder + '/labels.txt', 'w+') as file:
        for label in test_labels:
            file.write(str(label) + '\n')

    

