import random, json
import os



def generate_math_dataset(filename, dir, num_examples=2000):
    """Generate a dataset of simple math problems."""
    ops = [('+', lambda x, y: x + y), ('-', lambda x, y: x - y), ('*', lambda x, y: x * y), ('/', lambda x, y: x // y)]
    if not os.path.exists(dir):
        os.makedirs(dir)
    examples = []
    for _ in range(2000):
        x, y = random.randint(-1000, 1000), random.randint(0, 1000)
        op, fn = random.choice(ops)
        if op == '/' and y == 0:
            continue
        question = f"What is {x} {op} {y}?"
        answer = fn(x, y)
        examples.append({"input": question, "answer": answer})

    with open(os.path.join(dir, filename), "w") as f:
        json.dump(examples, f, indent=2)

if __name__== "__main__":
    generate_math_dataset("math_dataset_val.json", "data", 250)
