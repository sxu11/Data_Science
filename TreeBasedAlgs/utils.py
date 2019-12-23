
def calcSquaredError(predicts, actual):
    """

    :param predicts: Numpy array
    :param actual: number
    :return:
    """
    numerify = [0 if x == actual else 1 for x in predicts]

    return sum(numerify)

"""
Input: two numbers (test if a>=b); two object (test if a==b)
Output: bool True/False
"""

def ask(a, b):
    if isinstance(a, (np.int64, int, float)) and isinstance(b, (np.int64, int, float)):
        return ask_numeric(a, b)
    elif isinstance(a, str) and isinstance(b, str):
        return ask_categorical(a, b)
    else:
        raise Exception("ask not implemented for %s and %s!" % (type(a), type(b)))

def ask_numeric(a, b):
    return a >= b

def ask_categorical(a, b):
    return a == b


def test_question():
    # print question("a", "a")
    pass