import random
import os
import string


def generate_random_string(length=8):
    """
    Generate a random string of fixed length.
    """
    return "".join(random.choices(string.ascii_letters, k=length))


def generate_random_names(data_dir, file_name, seed):
    """
    Generates random names for entities and relations and splits relations.txt into train, valid, and test files.
    """
    # Read the input file
    with open(file_name, "r") as f:
        lines = f.readlines()

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Get the total number of triplets
    total_triplets = int(lines[0].strip())
    triplets = [line.strip().split() for line in lines[1:]]

    # Extract unique entities and relations
    entities = set()
    relations = set()
    for relation, entity1, entity2 in triplets:
        entities.add(entity1)
        entities.add(entity2)
        relations.add(relation)

    # Generate random names for entities and relations
    entity_list = sorted(list(entities))
    relation_list = sorted(list(relations))
    entity_to_random_name = {entity: generate_random_string() for entity in entity_list}
    relation_to_random_name = {
        relation: generate_random_string(length=4) for relation in relation_list
    }

    # Generate unique IDs for entities and relations
    entity_to_id = {entity: idx for idx, entity in enumerate(entity_list)}
    relation_to_id = {relation: idx for idx, relation in enumerate(relation_list)}

    # Write entity2id.txt
    with open(os.path.join(data_dir, "entity2id.txt"), "w") as f:
        f.write(f"{len(entity_list)}\n")
        for entity, idx in entity_to_id.items():
            f.write(f"{entity_to_random_name[entity]} {idx}\n")

    # Write relation2id.txt
    with open(os.path.join(data_dir, "relation2id.txt"), "w") as f:
        f.write(f"{len(relation_list)}\n")
        for relation, idx in relation_to_id.items():
            f.write(f"{relation_to_random_name[relation]} {idx}\n")

    # Shuffle and split the data
    random.seed(seed)
    random.shuffle(triplets)
    train_end = int(0.8 * total_triplets)
    valid_end = train_end + int(0.1 * total_triplets)

    train_triplets = triplets[:train_end]
    valid_triplets = triplets[train_end:valid_end]
    test_triplets = triplets[valid_end:]

    # Function to write split files
    def write_split_file(file_name, triplets, data_dir):
        with open(os.path.join(data_dir, file_name), "w") as f:
            f.write(f"{len(triplets)}\n")
            for relation, entity1, entity2 in triplets:
                f.write(
                    f"{entity_to_id[entity1]} {entity_to_id[entity2]} {relation_to_id[relation]}\n"
                )

    # Write train2id.txt, valid2id.txt, and test2id.txt
    write_split_file("train2id.txt", train_triplets, data_dir)
    write_split_file("valid2id.txt", valid_triplets, data_dir)
    write_split_file("test2id.txt", test_triplets, data_dir)


# Generate data for 20 different seeds
random_seeds = random.sample(range(1000, 10000), 20)
with open("random_seeds.txt", "w") as f:
    f.write("\n".join(map(str, random_seeds)))
for i, seed in enumerate(random_seeds):
    data_dir = f"./data/data_split{i}"
    generate_random_names(data_dir, "relations.txt", seed)
