"""Безопасная загрузка функции детектирования из сгенерированной программы.

Код LLM не импортируется как модуль: из AST берётся только функция detect.
Разрешены условия, простая арифметика и чтение известных сетевых признаков.
Кандидат интерпретируется как ограниченный числовой язык без exec и eval.
"""

import ast
import math
import operator
from pathlib import Path

from traffic_data.dataset_real import feature_names

MAX_SOURCE_CHARS = 12000
MAX_AST_NODES = 230
ALLOWED_CALLS = {"int": int, "float": float, "abs": abs, "min": min,
                 "max": max, "round": round, "bool": bool}
ALLOWED_NODES = (
    ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.If, ast.Assign,
    ast.AugAssign, ast.Expr, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Subscript, ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Compare,
    ast.IfExp, ast.Call, ast.Attribute, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.And, ast.Or, ast.Not, ast.USub,
    ast.UAdd, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def validate_detector(source: str) -> ast.FunctionDef:
    """Находит detect(event) и запрещает операции вне безопасного подмножества Python."""
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError("Слишком большой кандидат")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "detect"]
    if len(functions) != 1:
        raise ValueError("Нужна ровно одна функция detect(event)")
    func = functions[0]
    if (len(func.args.args) != 1 or func.args.args[0].arg != "event"
            or func.args.vararg or func.args.kwarg or func.args.kwonlyargs
            or func.args.defaults or func.decorator_list or func.args.posonlyargs):
        raise ValueError("Функция должна иметь сигнатуру detect(event)")
    # Докстринг не выполняет никаких действий. Длинные пояснения от LLM
    # допустимы, но остальные строковые константы по-прежнему ограничены.
    docstring_node = None
    if func.body and isinstance(func.body[0], ast.Expr):
        first = func.body[0].value
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            docstring_node = first
    nodes = list(ast.walk(func))
    if len(nodes) > MAX_AST_NODES:
        raise ValueError("Слишком сложный кандидат")
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node is not func:
            raise ValueError("Вложенные функции запрещены")
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Недопустимая конструкция: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise ValueError("Запрещены служебные имена")
            if isinstance(node.ctx, ast.Store) and node.id in ({"event"} | set(ALLOWED_CALLS)):
                raise ValueError("Недопустимое присваивание")
        if isinstance(node, ast.Subscript):
            if not isinstance(node.ctx, ast.Load):
                raise ValueError("Изменение входного события запрещено")
            if not (isinstance(node.value, ast.Name) and node.value.id == "event"
                    and isinstance(node.slice, ast.Constant) and node.slice.value in feature_names()):
                raise ValueError("Допускается только чтение разрешённых полей event")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "event" and node.attr == "get"):
                raise ValueError("Доступ к атрибутам запрещён")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_CALLS:
                    raise ValueError("Недопустимый вызов функции")
            elif not (isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
                raise ValueError("Недопустимый вызов метода")
            if isinstance(node.func, ast.Attribute):
                if not (1 <= len(node.args) <= 2 and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value in feature_names()):
                    raise ValueError("event.get принимает только известные поля")
            if node.keywords:
                raise ValueError("Именованные аргументы запрещены")
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow):
            raise ValueError("Возведение в степень запрещено")
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            if len(node.value) > 120 and node is not docstring_node:
                raise ValueError("Слишком длинная константа")
    return func


MAX_NUMBER = 10 ** 100
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


def scalar(value):
    """Ограничивает тип и величину каждого промежуточного значения."""
    if type(value) not in (int, float, bool) or abs(value) > MAX_NUMBER or not math.isfinite(value):
        raise ValueError("Разрешены только ограниченные конечные числа")
    return value


def interpret(func: ast.FunctionDef, event: dict, features: frozenset) -> int:
    """Выполняет только числовые выражения и конечные ветвления AST."""
    if type(event) is not dict:
        raise ValueError("Событие должно быть словарём")
    local = {}

    def expr(node):
        """Вычисляет выражение без доступа к объектам Python."""
        if isinstance(node, ast.Constant):
            return scalar(node.value)
        if isinstance(node, ast.Name) and node.id in local:
            return local[node.id]
        if isinstance(node, ast.Subscript):
            key = node.slice.value
            if key not in features:
                raise ValueError("Запрещённое поле")
            return scalar(event[key])
        if isinstance(node, ast.BinOp):
            return scalar(BINARY[type(node.op)](expr(node.left), expr(node.right)))
        if isinstance(node, ast.UnaryOp):
            value = expr(node.operand)
            return scalar(not value if isinstance(node.op, ast.Not) else
                          -value if isinstance(node.op, ast.USub) else +value)
        if isinstance(node, ast.BoolOp):
            value = expr(node.values[0])
            for item in node.values[1:]:
                if bool(value) != isinstance(node.op, ast.And):
                    break
                value = expr(item)
            return value
        if isinstance(node, ast.Compare):
            left = expr(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = expr(right_node)
                if not COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return expr(node.body if expr(node.test) else node.orelse)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                key = node.args[0].value
                if key not in features:
                    raise ValueError("Запрещённое поле")
                default = expr(node.args[1]) if len(node.args) == 2 else None
                return scalar(event.get(key, default))
            name = node.func.id
            args = [expr(arg) for arg in node.args]
            if name == "round" and len(args) == 2 and (type(args[1]) is not int or abs(args[1]) > 100):
                raise ValueError("Недопустимая точность округления")
            return scalar(ALLOWED_CALLS[name](*args))
        raise ValueError("Недопустимое числовое выражение")

    def block(statements):
        """Обрабатывает конечный блок и распространяет return из ветки."""
        for statement in statements:
            if isinstance(statement, ast.Return):
                return expr(statement.value)
            if isinstance(statement, ast.If):
                result = block(statement.body if expr(statement.test) else statement.orelse)
                if result is not None:
                    return result
            elif isinstance(statement, ast.Assign):
                value = expr(statement.value)
                for target in statement.targets:
                    if not isinstance(target, ast.Name):
                        raise ValueError("Присваивание только числовой переменной")
                    local[target.id] = value
            elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                name = statement.target.id
                local[name] = scalar(BINARY[type(statement.op)](local[name], expr(statement.value)))
            elif isinstance(statement, ast.Expr):
                if not (isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)):
                    expr(statement.value)
            else:
                raise ValueError("Недопустимая инструкция")
        return None

    value = block(func.body)
    if type(value) not in (int, bool) or value not in (0, 1):
        raise ValueError("detect должен возвращать 0 или 1")
    return int(value)


def load_detector(path: Path):
    """Загружает ограниченный AST; никакой кандидат не компилируется и не исполняется Python."""
    with path.open(encoding="utf-8") as source:
        func = validate_detector(source.read(MAX_SOURCE_CHARS + 1))
    features = frozenset(feature_names())

    def detector(event):
        """Проверяет вход и интерпретирует один сетевой поток."""
        try:
            return interpret(func, event, features)
        except (ArithmeticError, KeyError, TypeError) as error:
            raise ValueError(f"Ошибка числового правила: {error}") from error
    return detector


def predict(path: Path, events: list[dict]) -> list[int]:
    """Возвращает ответы проверенного детектора с ограничением числа событий."""
    if len(events) > 5000:
        raise ValueError("Слишком много событий")
    detector = load_detector(path)
    return [detector(event) for event in events]
