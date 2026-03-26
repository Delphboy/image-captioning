from typing import List


def preprocess_caption(caption: str) -> str:
    # Clean sentence list following: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf Section 4
    caption = caption.lower()

    # Disgard non-alphanumeric characters
    non_alphanumeric = [chr(i) for i in range(33, 128) if not chr(i).isalnum()]

    for char in non_alphanumeric:
        caption = caption.replace(char, "")
    while "  " in caption:
        caption = caption.replace("  ", " ")

    return caption.strip()


def preprocess_captions(captions: List[str]) -> List[str]:
    # Clean sentence list following: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf Section 4
    captions = [caption.lower() for caption in captions]

    # Disgard non-alphanumeric characters
    non_alphanumeric = [chr(i) for i in range(33, 128) if not chr(i).isalnum()]
    cleaned = []

    for sentence in captions:
        for char in non_alphanumeric:
            sentence = sentence.replace(char, "")
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")
        cleaned.append(sentence.strip())
    return cleaned
