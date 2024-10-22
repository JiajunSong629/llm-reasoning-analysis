def get_seqs_happy_negation(template):
    words = ["healthy", "happy", "kind"]
    names = ["Alice", "Bob", "Charlie"]
    instruction_templates = {
        0: "{{prompt}}",
        1: "For the following question, please answer Yes or No. {{prompt}}",
        2: "{{prompt}} The answer is",
        3: "The teacher asks '{{prompt}}'. The student answers",
    }
    instruction = instruction_templates[template]

    seqs = []
    for word in words:
        for name in names:
            seq = [
                instruction.replace(
                    "{{prompt}}", f"{name} is {word}. Is {name} {word}?"
                ),
                instruction.replace(
                    "{{prompt}}", f"{name} is un{word}. Is {name} {word}?"
                ),
                instruction.replace(
                    "{{prompt}}", f"{name} is not {word}. Is {name} {word}?"
                ),
                instruction.replace(
                    "{{prompt}}", f"{name} is not un{word}. Is {name} {word}?"
                ),
            ]
            seqs.append(seq)

    return seqs


def get_seqs_math(template):
    instruction_templates = {
        0: "{{prompt}}",
        1: "For the following question, please answer Yes or No. {{prompt}}",
        2: "{{prompt}} The answer is",
        3: "The teacher asks '{{prompt}}'. The student answers",
    }
    instruction = instruction_templates[template]

    seqs = []
    for num_pairs in [
        (9.11, 9.8),
        (1, 2),
        (0.11, 0.2),
    ]:
        n1, n2 = num_pairs
        seq = [
            instruction.replace("{{prompt}}", f"Is {n1} < {n2}?"),
            instruction.replace("{{prompt}}", f"Is {n1} smaller than {n2}?"),
            instruction.replace("{{prompt}}", f"Is {n1} - {n2} < 0?"),
            instruction.replace("{{prompt}}", f"Is {n1} > {n2}?"),
            instruction.replace("{{prompt}}", f"Is {n1} greater than {n2}?"),
            instruction.replace("{{prompt}}", f"Is {n1} - {n2} > 0?"),
        ]
        seqs.append(seq)

    return seqs
