import numpy as np

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--entity', type=int, default=50)
parser.add_argument('--relation', type=int, default=10)
parser.add_argument('--lambda1', type=float, default=2.0)
parser.add_argument('--lambda2', type=float, default=0.5)
parser.add_argument('--known', type=int, default=10)
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--output', type=str, default='relations.txt')

args = parser.parse_args()

np.random.seed(args.seed)

def generate_rule():
    total_triplets = np.random.poisson(args.lambda2) + 1
    total_addition_entity = np.random.randint(0, min(total_triplets, 2 * total_triplets - 2) + 1)
    total_relation = np.random.randint(1, min(total_triplets, args.relation) + 1)
    relations = np.random.choice(args.relation, total_relation, replace=False)
    Relation = np.random.choice(relations, total_triplets, replace=True)
    Entity = np.concatenate([np.random.choice(total_addition_entity + 2, 2 * total_triplets - 2, replace=True), np.array([0, 1])]).reshape(-1, 2)
    Rule = np.concatenate([Relation.reshape(-1, 1), Entity], axis=1)
    return Rule

def generate_data(i):
    total_rule = np.random.poisson(args.lambda1) + 1
    Rules = [(i, generate_rule()) for _ in range(total_rule)]
    return Rules

All_Rules = []
for i in range(args.relation):
    All_Rules += generate_data(i)
print(f"Generated {len(All_Rules)} rules")

known_relations = set()
for _ in range(args.known):
    known_relations.add((np.random.randint(args.relation), np.random.randint(args.entity), np.random.randint(args.entity)))

# Helper function to check if a given mapping satisfies all relations in the rule
def check_mapping(rule, mapping):
    for relation, entity1, entity2 in rule:
        real_entity1 = mapping.get(entity1, None)
        real_entity2 = mapping.get(entity2, None)
        if (relation, real_entity1, real_entity2) not in known_relations:
            return False
    return True

# Helper function for backtracking to try all possible mappings
def try_mapping(rule, mapping, pseudo_entities, idx=0):
    if idx == len(pseudo_entities):
        # If we've tried all pseudo entities and found a valid mapping, check the rule
        return check_mapping(rule, mapping)
    
    # Try assigning each possible real entity to the current pseudo entity
    pseudo_entity = pseudo_entities[idx]
    for entity in range(args.entity):
        # if entity not in mapping.values():  # Ensure we do not reuse an entity
        mapping[pseudo_entity] = entity
        if try_mapping(rule, mapping, pseudo_entities, idx + 1):
            return True
        del mapping[pseudo_entity]  # Backtrack
    return False

# deduce all known relations
i = 0
while True:
    print(f"Round {i}")
    i += 1
    new_relations = set()
    for rule_index, Rule in All_Rules:
        # Extract pseudo entities from the rule
        pseudo_entities = list(set(Rule[:, 1:3].flatten()))  # Pseudo entities are the entity columns in the rule

        # Try mapping pseudo entities to real entities
        mapping = {}
        if try_mapping(Rule, mapping, pseudo_entities):
            # If a valid mapping is found, deduce the real relations
            # Create the new relation using the mapping of pseudo entities to real ones
            real_entity1 = mapping.get(0, None)
            real_entity2 = mapping.get(1, None)
            if real_entity1 is not None and real_entity2 is not None:
                if (rule_index, real_entity1, real_entity2) not in known_relations:
                    new_relations.add((rule_index, real_entity1, real_entity2))  # Add the new relation

    print(f"Found {len(new_relations)} new relations")
    if len(new_relations) == 0:
        break
    known_relations.update(new_relations)

# Writing the relations to a file
with open(args.output, 'w') as file:
    # First line: the number of relations
    file.write(f"{len(known_relations)}\n")
    
    # Write each relation in the format (relation, entity1, entity2)
    for relation in known_relations:
        file.write(f"{relation[0]} {relation[1]} {relation[2]}\n")

print(f"Relations have been written to {args.output}")