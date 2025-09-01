import random, json
import os



def generate_math_dataset(filename, dir, num_examples=200000, already_generated={}):
    """Generate a dataset of simple math problems."""
    ops = [('+', lambda x, y: x + y), ('-', lambda x, y: x - y), ('*', lambda x, y: x * y), ('/', lambda x, y: x // y)]
    if not os.path.exists(dir):
        os.makedirs(dir)
    examples = []
    for _ in range(num_examples):
        x, y = None, None
        op, fn = None, None
        question = None
        answer = None
        while True:
            x, y = random.randint(-1000, 1000), random.randint(0, 1000)
            op, fn = random.choice(ops)
            if op == '/' and y == 0:
                continue
            question = f"What is {x} {op} {y}?"
            answer = fn(x, y)
            if (x, y, op) in already_generated:
                continue
            break
        already_generated[(x, y, op)] = True
        examples.append({"input": question, "answer": answer})

    with open(os.path.join(dir, filename), "w") as f:
        json.dump(examples, f, indent=2)
    return already_generated

if __name__== "__main__":
    already_generated = generate_math_dataset("math_dataset.json", "data", 200000, {})
    generate_math_dataset("math_dataset_val.json", "data", 10000, already_generated)
    print("Done")
