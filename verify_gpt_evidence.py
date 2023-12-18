import json
import os
import re


FOLDER_PATHS = ["test_gpt", "train_gpt15k", "train_gpt_TOEFL", "test_gpt_TOEFL"]


def is_substring_with_letters(a, b):
    """
    Identify is one string is a substring of another one
    """
    a_letters = re.sub(r'[^a-zA-Z]', '', a)
    b_letters = re.sub(r'[^a-zA-Z]', '', b)


    return b_letters in a_letters


if __name__ == "__main__":
    for folder_path in FOLDER_PATHS:
        verified = {}
        hallucination = 0
        correct = 0
        file_error = 0

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as file:
                        content = json.load(file)

                    for key, value in content.items():
                        v = value[0].replace("EVIDENCE:", "").replace('"', '').strip()
                        match = is_substring_with_letters(key, v)
                        if match:
                            verified[key] = v
                            correct += 1
                        else:
                            hallucination += 1
                except:
                    file_error +=1

        print("Correct: ", correct, "Hallucinations: ", hallucination)
        print(file_error)

        with open(f'{folder_path}_verified.json', 'w') as json_file:
            json.dump(verified, json_file)

