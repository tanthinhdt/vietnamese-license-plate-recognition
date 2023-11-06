import math


def compute_linear_equation(x1: int, y1: int, x2: int, y2: int) -> tuple:
    """
    Given 2 points, compute the linear equation.
    :param x1:  The x-coordinate of the first point.
    :param y1:  The y-coordinate of the first point.
    :param x2:  The x-coordinate of the second point.
    :param y2:  The y-coordinate of the second point.
    :return:    The linear equation.
    """
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b


def check_point_closeness(x: int, y: int, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Check whether a point is close to a line.
    :param x:   The x-coordinate of the point.
    :param y:   The y-coordinate of the point.
    :param x1:  The x-coordinate of the first point of the line.
    :param y1:  The y-coordinate of the first point of the line.
    :param x2:  The x-coordinate of the second point of the line.
    :param y2:  The y-coordinate of the second point of the line.
    :return:    Whether the point is close to the line.
    """
    a, b = compute_linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=3)
