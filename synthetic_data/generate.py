import numpy as np
from numba import jit
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()

parser.add_argument('--entity', type=int, default=300)
parser.add_argument('--relation', type=int, default=30)
parser.add_argument('--lambda1', type=float, default=1)
parser.add_argument('--lambda2', type=float, default=1)
parser.add_argument('--known', type=int, default=200)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--output_rule', type=str, default='rules.txt')
parser.add_argument('--output_relation', type=str, default='relations.txt')

args = parser.parse_args()

np.random.seed(args.seed)

def generate_rule():
    total_triplets = np.random.poisson(args.lambda2) + 1
    total_addition_entity = min(np.random.choice(3, p=[0.2, 0.3, 0.5]), total_triplets - 1)
    total_relation = np.random.randint(1, min(total_triplets, args.relation) + 1)
    relations = np.random.choice(args.relation, total_relation, replace=False)
    Relation = np.random.choice(relations, total_triplets, replace=True)
    before_shuffle = np.concatenate([np.random.choice(total_addition_entity + 2, 2 * total_triplets - total_addition_entity - 2, replace=True), np.arange(total_addition_entity + 2)])
    np.random.shuffle(before_shuffle)
    Entity = before_shuffle.reshape(-1, 2)
    if np.sum(Entity[:, 0] == Entity[:, 1]) > 0:
        return generate_rule()
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
    entities = np.random.choice(args.entity, 2, replace=False)
    known_relations.add((i, entities[0], entities[1]))

for _ in range(args.known - args.relation):
    entities = np.random.choice(args.entity, 2, replace=False)
    known_relations.add((np.random.randint(args.relation), entities[0], entities[1]))
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
def try_mapping(rule, knowns, mapping, number_pseudo_entities, rule_index, idx=0):
    if idx == number_pseudo_entities:
        # If we've tried all pseudo entities and found a valid mapping, check the rule
        return check_mapping(rule, mapping, knowns)
    for entity in range(number):
        if idx == 1 and entity == mapping[0]:
            continue
        if idx == 1 and (rule_index, mapping[0], entity) in knowns:
            continue
        mapping[idx] = entity
        answer = try_mapping(rule, knowns, mapping, number_pseudo_entities, rule_index, idx + 1)
        if answer:
            return True
    return False

@jit(nopython=True)
def get_all_possible(rule, knowns, mapping, number_pseudo_entities, rule_index, idx=0):
    Answer_list = []
    for i in range(number):
        mapping[0] = i
        for j in range(number):
            if i == j or (rule_index, i, j) in knowns:
                continue
            mapping[1] = j
            answer = try_mapping(rule, knowns, mapping, number_pseudo_entities, rule_index, 2)
            if answer:
                Answer_list.append((i, j))
    return Answer_list

# deduce all known relations
i = 0
# unused_relation = known_relations.copy()
used_rule = np.zeros(len(All_Rules))
while True:
    print(f"Round {i}")
    i += 1
    new_relations = set()
    for j, (rule_index, Rule) in enumerate(tqdm(All_Rules)):
        # Extract pseudo entities from the rule
        number_pseudo_entities = len(set(Rule[:, 1:3].flatten()))

        # Try mapping pseudo entities to real entities
        mapping = np.zeros(number_pseudo_entities, dtype=np.int64) # Initialize the mapping
        answers = get_all_possible(Rule, known_relations, mapping, number_pseudo_entities, rule_index)
        if answers:
            # If a valid mapping is found, deduce the real relations
            # Create the new relation using the mapping of pseudo entities to real ones
            for answer in answers:
                # unused_relation = unused_relation - set(used_relation)
                new_relations.add((rule_index, answer[0], answer[1]))  # Add the new relation
                used_rule[j] = 1
    
    # unused_relation = unused_relation - set(new_relations)

    print(f"Found {len(new_relations)} new relations")
    if len(new_relations) == 0:
        break
    known_relations.update(new_relations)

# print(f"unused relations:", len(unused_relation))
print(f"used rule:", np.sum(used_rule))
with open(args.output_rule, 'w') as file:
    for j, (i, Rule) in enumerate(All_Rules):
        if used_rule[j] != 1:
            continue
        file.write(f"{i}\n")
        for row in Rule:
            file.write(" ".join(map(str, row)) + "\n")


# known_relations = known_relations - unused_relation

known_entities = set()
for relation in known_relations:
    known_entities.add(relation[1])
    known_entities.add(relation[2])
known_entities = list(known_entities)
entity_mapping = {entity: i for i, entity in enumerate(known_entities)}
known_relations = {(relation[0], entity_mapping[relation[1]], entity_mapping[relation[2]]) for relation in known_relations}
print(f"Generated {len(known_entities)} entities")

# Writing the relations to a file
with open(args.output_relation, 'w') as file:
    # First line: the number of relations
    file.write(f"{len(known_relations)}\n")
    
    # Write each relation in the format (relation, entity1, entity2)
    for relation in known_relations:
        file.write(f"{relation[0]} {relation[1]} {relation[2]}\n")

print(f"Relations have been written to {args.output_relation}")