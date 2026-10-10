from torchvision.datasets import CIFAR10
from transformers import CLIPImageProcessor # resize plus applying clip normalization to the dataset
import random
from sklearn.model_selection import train_test_split
from datasets import load_dataset


def load_dataset():

    # starting off with loading the full dataset
    full_train = CIFAR10(root='./data', train=True, download=True)
    full_test = CIFAR10(root='./data', train=False, download=True)

    # startifying the dataset based off the instructions in the assignment
    train_indices = train_test_split(range(len(full_train)), train_size=2000, stratify=full_train.targets, random_state=42)
    test_indices = train_test_split(range(len(full_test)), test_size=500, stratify=full_test.targets, random_state=42)

    # train test split returns 2 indices, selected indices and 'removed' indices
    # by using [0] we are getting the selected indices

    train_indices = train_indices[0]
    test_indices = test_indices[1]

    # Caption Dataset
    caption_templates = [
        "A photo of a {}.",
        "This is a {}.",
        "An image of a {}.",
        "A {} is shown here.",
        "This image shows a {}.",
        "This is an example of a {}.",
    ]

    #Applying these to images in a rotating fashion

    #checking classes of cifar 10
    cifar_10_classes = full_train.classes
    print(cifar_10_classes)


    captions = []
    for i, index in enumerate(train_indices):
        # train_indices contains the indices that have been selected
        # we are getting the actual image against the target index and its label
        label = full_train.targets[index]
        # label is 0 - 9, getting word against it
        class_name = cifar_10_classes[label]
        template = caption_templates[i % len(caption_templates)] # This ensures that we keep on cycling
        captions.append({
            "image_idx": index,
            "label": label,
            "caption": template.format(class_name)
            })


    random.seed(42)

    vqa_train = []
    for index in train_indices:
        label = full_train.targets[index]
        class_name = cifar_10_classes[label]

        # Q1: Recognition
        vqa_train.append({"image_idx": index, "question": "What object is shown?", "answer": class_name, "skill": "recognition"})

        # Q2: Binary
        if random.random() < 0.5:
            # Positive: ask about the correct class
            vqa_train.append({"image_idx": index, "question": f"Is there a {class_name}?", "answer": "yes", "skill": "binary"})
        else:
            # Negative: ask about a random wrong class
            # random choice will choose one out of the nine possibilities at random
            wrong_class = random.choice([c for c in cifar_10_classes if c != class_name])
            vqa_train.append({"image_idx": index, "question": f"Is there a {wrong_class}?", "answer": "no", "skill": "binary"})

        # Q3: Abstraction
        answer = "vehicle" if class_name in ["airplane", "automobile", "ship", "truck"] else "living"
        vqa_train.append({"image_idx": index, "question": "Vehicle or living thing?", "answer": answer, "skill": "abstraction"})

        # Q4: Attribute
        answer = "yes" if class_name in ["airplane", "bird"] else "no"
        vqa_train.append({"image_idx": index, "question": "Can it fly?", "answer": answer, "skill": "attribute"})

        # Q5: Category
        answer = "yes" if class_name in ["bird", "cat", "deer", "dog", "frog", "horse"] else "no"
        vqa_train.append({"image_idx": index, "question": "Is this an animal?", "answer": answer, "skill": "category"})

    print(len(vqa_train))

    vqa_val = []
    for index in test_indices:
        label = full_test.targets[index]
        class_name = cifar_10_classes[label]

        # Q1: Recognition
        vqa_val.append({"image_idx": index, "question": "What object is shown?", "answer": class_name, "skill": "recognition"})

        # Q2: Binary
        if random.random() < 0.5:
            vqa_val.append({"image_idx": index, "question": f"Is there a {class_name}?", "answer": "yes", "skill": "binary"})
        else:
            wrong_class = random.choice([c for c in cifar_10_classes if c != class_name])
            vqa_val.append({"image_idx": index, "question": f"Is there a {wrong_class}?", "answer": "no", "skill": "binary"})

        # Q3: Abstraction
        answer = "vehicle" if class_name in ["airplane", "automobile", "ship", "truck"] else "living"
        vqa_val.append({"image_idx": index, "question": "Vehicle or living thing?", "answer": answer, "skill": "abstraction"})

        # Q4: Attribute
        answer = "yes" if class_name in ["airplane", "bird"] else "no"
        vqa_val.append({"image_idx": index, "question": "Can it fly?", "answer": answer, "skill": "attribute"})

        # Q5: Category
        answer = "yes" if class_name in ["bird", "cat", "deer", "dog", "frog", "horse"] else "no"
        vqa_val.append({"image_idx": index, "question": "Is this an animal?", "answer": answer, "skill": "category"})

    print(len(vqa_val))


    #I am loading a dataset to measure baseline perplexity on smollm.
    #We will measure baseline before our training and after our training to ensure our training doesn't cause catastrophic forgetting

    alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_subset = alpaca_dataset.select(range(1000))

    print(alpaca_subset[0])



    return (train_indices, test_indices, captions, cifar_10_classes, vqa_train, vqa_val, full_train, full_test, alpaca_dataset, alpaca_subset)

def load_clip_processor():

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(processor.image_mean)
    print(processor.image_std)
    return processor











