import random
import json
import argparse


def generate_eval_set(num_per_category, seed, output_path):
    random.seed(seed)

    names = ["Alice", "Bob", "Tom", "Jack", "Emma", "Lucy",
             "Liam", "Noah", "Mia", "Olivia", "Ethan", "Ava"]
    animals = ["dog", "cat", "rabbit", "fox", "bear", "bird"]
    places = ["forest", "park", "school", "beach", "village"]
    objects = ["ball", "map", "key", "book", "box", "treasure"]
    causal_starters = [
        "because it was raining", "because he forgot his homework",
        "because the lights went out", "because she heard a strange noise"
    ]

    eval_set = []

    # 1️. Repetition
    rep_templates = [
        "Once upon a time, there was a {animal} who",
        "In a small {place}, a {animal} discovered a {object} that",
        "Long ago, a {animal} wanted to",
        "One quiet night in the {place}, something strange",
        "A curious {animal} found a mysterious {object} and"
    ]

    for _ in range(num_per_category):
        template = random.choice(rep_templates)
        prompt = template.format(
            animal=random.choice(animals),
            place=random.choice(places),
            object=random.choice(objects)
        )

        eval_set.append({
            "category": "repetition",
            "prompt": prompt,
            "expected_entities": []
        })

    # 2️. Entity
    entity_templates = [
        "{n1} and {n2} went to the {place} to find a {object}.",
        "{n1}, {n2}, and {n3} were playing in the {place}.",
        "At school, {n1} told {n2} about a {object}.",
        "{n1} helped {n2} after they lost a {object}."
    ]

    for _ in range(num_per_category):
        template = random.choice(entity_templates)

        num_names = random.choice([2, 3])
        sampled = random.sample(names, num_names)

        while len(sampled) < 3:
            sampled.append(None)

        prompt = template.format(
            n1=sampled[0],
            n2=sampled[1],
            n3=sampled[2] if sampled[2] is not None else sampled[0],
            place=random.choice(places),
            object=random.choice(objects)
        )

        expected = [n for n in sampled if n is not None]

        eval_set.append({
            "category": "entity",
            "prompt": prompt,
            "expected_entities": expected
        })

    # 3️. Structure
    structure_templates = [
        "Once upon a time, there was a {animal}.",
        "In a small {place}, a {animal} lived with his family.",
        "{n1} and {n2} were best friends.",
        "It was a cold day in the {place}."
    ]

    for _ in range(num_per_category):
        template = random.choice(structure_templates)

        prompt = template.format(
            animal=random.choice(animals),
            place=random.choice(places),
            n1=random.choice(names),
            n2=random.choice(names)
        )

        eval_set.append({
            "category": "structure",
            "prompt": prompt,
            "expected_entities": []
        })

    # 4️. Coherence
    coherence_templates = [
        "{n1} was late for school {cause}, so",
        "A storm started in the {place}, and",
        "{n1} lost a {object}, then",
        "After {n1} found the {object},"
    ]

    for _ in range(num_per_category):
        template = random.choice(coherence_templates)

        prompt = template.format(
            n1=random.choice(names),
            place=random.choice(places),
            object=random.choice(objects),
            cause=random.choice(causal_starters)
        )

        eval_set.append({
            "category": "coherence",
            "prompt": prompt,
            "expected_entities": []
        })

    random.shuffle(eval_set)

    with open(output_path, "w") as f:
        json.dump(eval_set, f, indent=2)

    print(f"Generated {len(eval_set)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_per_category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_set.json")

    args = parser.parse_args()

    generate_eval_set(args.num_per_category, args.seed, args.output)


if __name__ == "__main__":
    main()