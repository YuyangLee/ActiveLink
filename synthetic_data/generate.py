import numpy as np
from numba import jit
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--entity', type=int, default=100)
parser.add_argument('--relation', type=int, default=40)
parser.add_argument('--lambda1', type=float, default=2.0)
parser.add_argument('--lambda2', type=float, default=0.8)
parser.add_argument('--known', type=int, default=100)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--output_rule', type=str, default='rules.txt')
parser.add_argument('--output_relation', type=str, default='relations.txt')

args = parser.parse_args()

np.random.seed(args.seed)

def generate_rule():
    total_triplets = np.random.poisson(args.lambda2) + 1
    total_addition_entity = min(np.random.randint(0, min(total_triplets, 2 * total_triplets - 2) + 1), 3)
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
with open(args.output_rule, 'w') as file:
    for i, Rule in All_Rules:
        file.write(f"{i}\n")
        for row in Rule:
            file.write(" ".join(map(str, row)) + "\n")

known_relations = set()
for i in range(args.relation):
    known_relations.add((i, np.random.randint(args.entity), np.random.randint(args.entity)))

for _ in range(args.known - args.relation):
    known_relations.add((np.random.randint(args.relation), np.random.randint(args.entity), np.random.randint(args.entity)))
# Delete unused entities and rename the entities to be continuous
known_entities = set()
for relation in known_relations:
    known_entities.add(relation[1])
    known_entities.add(relation[2])
known_entities = list(known_entities)
entity_mapping = {entity: i for i, entity in enumerate(known_entities)}
known_relations = {(relation[0], entity_mapping[relation[1]], entity_mapping[relation[2]]) for relation in known_relations}
print(f"Generated {len(known_entities)} entities")
number = len(known_entities)

# Helper function to check if a given mapping satisfies all relations in the rule
@jit(nopython=True)
def check_mapping(rule, mapping, knowns):
    for relation, entity1, entity2 in rule:
        real_entity1 = mapping[entity1]
        real_entity2 = mapping[entity2]
        if (relation, real_entity1, real_entity2) not in knowns:
            return False
    return True

# Helper function for backtracking to try all possible mappings
@jit(nopython=True)
def try_mapping(rule, knowns, mapping, number_pseudo_entities, idx=0):
    if idx == number_pseudo_entities:
        # If we've tried all pseudo entities and found a valid mapping, check the rule
        return check_mapping(rule, mapping, knowns)

    for entity in range(number):
        mapping[idx] = entity
        if try_mapping(rule, knowns, mapping, number_pseudo_entities, idx + 1):
            return True
    return False

# deduce all known relations
i = 0
while True:
    print(f"Round {i}")
    i += 1
    new_relations = set()
    for rule_index, Rule in All_Rules:
        # Extract pseudo entities from the rule
        number_pseudo_entities = len(set(Rule[:, 1:3].flatten()))

        # Try mapping pseudo entities to real entities
        mapping = np.zeros(number_pseudo_entities, dtype=np.int64) # Initialize the mapping
        if try_mapping(Rule, known_relations, mapping, number_pseudo_entities):
            # If a valid mapping is found, deduce the real relations
            # Create the new relation using the mapping of pseudo entities to real ones
            real_entity1 = mapping[0]
            real_entity2 = mapping[1]
            if real_entity1 is not None and real_entity2 is not None:
                if (rule_index, real_entity1, real_entity2) not in known_relations:
                    new_relations.add((rule_index, real_entity1, real_entity2))  # Add the new relation

    print(f"Found {len(new_relations)} new relations")
    if len(new_relations) == 0:
        break
    known_relations.update(new_relations)

# Writing the relations to a file
with open(args.output_relation, 'w') as file:
    # First line: the number of relations
    file.write(f"{len(known_relations)}\n")
    
    # Write each relation in the format (relation, entity1, entity2)
    for relation in known_relations:
        file.write(f"{relation[0]} {relation[1]} {relation[2]}\n")

print(f"Relations have been written to {args.output_relation}")