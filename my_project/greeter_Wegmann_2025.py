def separate_names(names):
    normal_names = []
    upper_names = []
    for name in names:
        if name.isupper():
            upper_names.append(name)
        else:
            normal_names.append(name)
    return normal_names, upper_names


def create_normal_name_list(names):
    if len(names) == 1:
        return names[0]
    elif len(names) == 2:
        return f"{names[0]} and {names[1]}"
    else:
        names_without_last = names[:-1]
        komma_part = ", ".join(names_without_last)
        return f"{komma_part}, and {names[-1]}"


def create_upper_name_list(names):
    if len(names) == 1:
        return names[0]
    elif len(names) == 2:
        return f"{names[0]} AND {names[1]}"
    else:
        names_without_last = names[:-1]
        komma_part = ", ".join(names_without_last)
        return f"{komma_part}, AND {names[-1]}"


def flatten_list(name_list):
    flattend_list = []
    for name in name_list:
        if name.startswith('"'):
            # Die " am Anfang und am Ende der Zeichenkette wegschneiden
            flattend_list.append(name[1:-1])
        elif "," in name:
            flattend_list.extend([entry.strip() for entry in name.split(",")])
        else:
            flattend_list.append(name)
    return flattend_list


def greet(name_or_list):
    if name_or_list is None:
        return "Hello, my friend."
    elif isinstance(name_or_list, list):
        name_or_list = flatten_list(name_or_list)
        normal_names, upper_names = separate_names(name_or_list)
        if len(upper_names) == 0:
            return f"Hello, {create_normal_name_list(normal_names)}."
        else:
            return f"Hello, {create_normal_name_list(normal_names)}. AND HELLO {create_upper_name_list(upper_names)}!"
    elif name_or_list.isupper():
        return f"HELLO, {name_or_list}!"
    return f"Hello, {name_or_list}."


if __name__ == "__main__":
    print(greet("Bob"))
