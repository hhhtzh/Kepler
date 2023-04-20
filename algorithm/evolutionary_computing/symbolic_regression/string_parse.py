import re
from algorithm.evolutionary_computing.symbolic_regression.operator_definitions \
    import INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION, MULTIPLICATION, \
           DIVISION, SIN, COS, SINH, COSH, EXPONENTIAL, LOGARITHM, POWER, ABS, \
           SQRT
import numpy as np
operators = {"+", "-", "*", "/", "^"}
functions = {"sin", "cos", "sinh", "cosh", "exp", "log", "abs", "sqrt"}
precedence = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
operator_map = {"+": ADDITION, "-": SUBTRACTION, "*": MULTIPLICATION,
                "/": DIVISION, "^": POWER, "X": VARIABLE, "x": VARIABLE,
                "C": CONSTANT, "c": CONSTANT,
                "sin": SIN, "cos": COS, "sinh": SINH, "cosh": COSH,
                "exp": EXPONENTIAL, "log": LOGARITHM, "abs": ABS,
                "sqrt": SQRT}
var_or_const_pattern = re.compile(r"([XC])_(\d+)", re.IGNORECASE)
int_pattern = re.compile(r"\d+")
negative_pattern = re.compile(r"-([^\s\d])")  # matches -N where N = non-number
non_unary_op_pattern = re.compile(r"([*/^()])")  # matches *, /, ^, (, or )
#r取消转义

def eq_string_to_infix_tokens(eq_string):
    """Converts an equation string to infix_tokens

    Parameters
    ----------
    eq_string : str
        A string corresponding to an equation

    Returns
    -------
    infix_tokens : list of str

        A list of string tokens that correspond
        to the expression given by eq_string
        返回一个存储中缀符号式的列表
    """
    if any(bad_token in eq_string for bad_token in ["zoo", "I", "oo",
                                                       "nan"]):
        raise RuntimeError("包含词无法分析")
    #
    eq_string = eq_string.replace(")(", ")*(").replace("**", "^")
    #替换符号

    eq_string = negative_pattern.sub(r"-1 * \1", eq_string)
    # replace -token with -1.0 * token if token != a number
    #当负号后的不是数字的时候替换为乘负1
    tokens = non_unary_op_pattern.sub(r" \1 ", eq_string).split(" ")
    #用空格替换掉符号，然后在空格处切分
    tokens = [x.lower() for x in tokens if x != ""]
    #把符号的字母都变为小写
    return tokens
    #返回一个存储着string类型的符号名的数组

def postfix_to_command_array_and_constants(postfix_tokens):
    """Converts a list of postfix tokens to its corresponding command array
    and list of constants
    把一个前缀表达式转化为它对应的操作符队列和常量

    Parameters
    ----------
    postfix_tokens : list of str
        A list of postfix string tokens

    Returns
    -------
    command_array, constants : Nx3 numpy array of int, list of numeric
    返回numpy类型的数组
        A command array and list of constants
        corresponding to the expression given by the postfix_tokens
    """
    stack = []  # index -1 = top (the data structure, not a command array)
    command_array = []
    i = 0
    command_to_i = {}
    constants = []
    n_constants = 0

    for token in postfix_tokens:
        if token in operators:
            #如果是操作符出栈两个数作运算
            operands = stack.pop(), stack.pop()
            command = [operator_map[token], operands[1], operands[0]]
        elif token in functions:
            #如果是函数出栈一个数作运算
            operand = stack.pop()
            command = [operator_map[token], operand, operand]
        else:
            var_or_const = var_or_const_pattern.fullmatch(token)
            integer = int_pattern.fullmatch(token)
            #分类常量变量和数字
            if var_or_const:
                groups = var_or_const.groups()
                command = [operator_map[groups[0]], int(groups[1]),
                           int(groups[1])]
            elif integer:
                operand = int(token)
                command = [INTEGER, operand, operand]
            else:
                try:
                    command = [CONSTANT, n_constants, n_constants]

                    constant = float(token)
                    constants.append(constant)
                    n_constants += 1
                except ValueError as err:
                    raise RuntimeError(f"未知符号 {token}") from err
        if tuple(command) in command_to_i:
            stack.append(command_to_i[tuple(command)])
        else:
            command_to_i[tuple(command)] = i
            command_array.append(command)
            stack.append(i)
            i += 1

    if len(stack) > 1:
        raise RuntimeError("前缀表达式生成错误")

    return np.array(command_array, dtype=int), constants


def eq_string_to_command_array_and_constants(eq_string):
    """Converts an equation string to its corresponding command
    array and list of constants

    Parameters
    ----------
    eq_string : str
        A string corresponding to an equation
        把字符串类型的方程转换为numpy数组类型

    Returns
    -------
    command_array, constants : Nx3 numpy array of int, list of numeric
        A command array and list of constants
        corresponding to the expression given by eq_string
    """
    infix_tokens = eq_string_to_infix_tokens(eq_string)
    #获得中缀符号表达式的列表
    postfix_tokens = infix_to_postfix(infix_tokens)
    #获得后缀
    return postfix_to_command_array_and_constants(postfix_tokens)

def infix_to_postfix(infix_tokens):
    """Converts a list of infix tokens into its corresponding
    list of postfix tokens (e.g. ["a", "+", "b"] -> ["a", "b", "+"])
    把中缀表达式转换为后缀表达式
    Based on the Dijkstra's Shunting-yard algorithm
    基于dijkstra的调度场算法
    Parameters
    ----------
    infix_tokens : list of str
        A list of infix string tokens

    Returns
    -------
    postfix_tokens : list of str
        A list of postfix string tokens corresponding
        to the expression given by infix_tokens
        返回存储字符串的列表的后缀表达式
    """
    stack = []  # index -1 = top (the data structure, not a command array)
    output = []
    for token in infix_tokens:
        if token in operators:
            while len(stack) > 0 and stack[-1] in operators and \
                (precedence[stack[-1]] > precedence[token] or
                 precedence[stack[-1]] == precedence[token] and token != "^"):
                output.append(stack.pop())
            stack.append(token)
        elif token == "(" or token in functions:
            stack.append(token)
        elif token == ")":
            while len(stack) > 0 and stack[-1] != "(":
                output.append(stack.pop())
            if len(stack) == 0 or stack.pop() != "(":  # get rid of "("
                raise RuntimeError("括号不匹配")
            if len(stack) > 0 and stack[-1] in functions:
                output.append(stack.pop())
        else:
            output.append(token)

    while len(stack) > 0:
        token = stack.pop()
        if token == "(":
            raise RuntimeError("括号不匹配")
        output.append(token)

    return output