import random
import json
import argparse

def main(output_path, num_samples=2000, seed=42):
    random.seed(seed)

    names = [
        "Alice", "Bob", "Tom", "Jack", "Emma", "Lucy",
        "Liam", "Noah", "Mia", "Olivia", "Ethan", "Ava",
        "Sophia", "James", "Henry", "Grace", "Ella", "Leo",
        "Chloe", "Max"
    ]
    places = [
        "forest", "park", "school", "beach", "village",
        "library", "museum", "playground", "zoo", "garden",
        "castle", "market", "river", "mountain", "cafe"
    ]
    objects = [
        "ball", "map", "key", "book", "box", "treasure",
        "letter", "puzzle", "magic wand", "hat", "coin",
        "paintbrush", "kite", "lantern", "toy"
    ]

    # Templates for 2 people
    templates_2p = [
        "{p1} showed {p2} how to use it. Then, {p1} and {p2} went to the {place} together.",
        "{p2} was surprised when {p1} shared a {obj}. They laughed and played with it.",
        "{p1} and {p2} discovered a {obj} in the {place}. They decided to explore together.",
        "{p2} looked at {p1} as they found a {obj} at the {place}. Together, they had fun.",
        "In the {place}, {p1} handed a {obj} to {p2}. They played until sunset."
    ]

    # Templates for 3 people
    templates_3p = [
        "{p1}, {p2}, and {p3} found a {obj} in the {place} and decided to share it. {p1} gave it to {p2}, then {p3} joined them.",
        "In the {place}, {p1}, {p2}, and {p3} discovered a {obj}. Together, they learned how to use it and had fun.",
        "{p1}, {p2}, and {p3} were curious about a {obj} in the {place}. After some adventures, they all became friends.",
        "At the {place}, {p3} noticed a {obj} and called {p1} and {p2} over. They played with it happily.",
        "While in the {place}, {p2} found a {obj}. {p1} and {p3} joined to explore and learn together."
    ]

    entity_patch_data = []

    for _ in range(num_samples):
        n_entities = random.choice([2, 3])
        sampled = random.sample(names, n_entities)
        random.shuffle(sampled)

        place = random.choice(places)
        obj = random.choice(objects)

        if n_entities == 2:
            p1, p2 = sampled
            prompt = f"At {place}, {p1} told {p2} about a {obj}."
            completion_template = random.choice(templates_2p)
            completion = completion_template.format(place=place, p1=p1, p2=p2, obj=obj)
        else:
            p1, p2, p3 = sampled
            prompt = f"{p1}, {p2}, and {p3} were playing in the {place}."
            completion_template = random.choice(templates_3p)
            completion = completion_template.format(place=place, p1=p1, p2=p2, p3=p3, obj=obj)

        entity_patch_data.append({
            "prompt": prompt,
            "completion": completion,
            "expected_entities": sampled
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entity_patch_data, f, indent=2, ensure_ascii=False)

    print(f"Generated Entity Patch with {len(entity_patch_data)} samples at '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Entity Patch Dataset")
    parser.add_argument("--output_path", type=str, default="entity_patch.json",
                        help="Path to save the entity patch JSON")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args.output_path, args.num_samples, args.seed)