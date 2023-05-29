from tabulate import tabulate
import warnings
import numpy as np
import math
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)


def u_2_equation(t, u_1, u_2):
    # ! CHANGE IN CASE FUNCTION DEFINITIONS IS DIFFERENT
    return 550 * math.cos(11 * t) - 5 * u_2 - 6 * u_1


def analytic_equation_q(t, iterations, step):
    evaluations = []
    for i in range(0, iterations):
        evaluations.append(
            [(-253 / 65) * math.cos(11 * t) + (121 / 65) * math.sin(11 * t) - (44 / 5) *
             math.exp(-2 * t) + (165 / 13) * math.exp(-3 * t)]
        )
        t += step

    return evaluations


def analytic_equation_i(t, iterations, step):
    evaluations = []
    for i in range(0, iterations):
        evaluations.append([
            (2783 / 65) * math.sin(11 * t) + (1331 / 65) * math.cos(11 * t) - (88 / 5) *
            math.exp(-2 * t) - (495 / 13) * math.exp(-3 * t)]
        )
        t += step

    return evaluations


def runge_kutta_fourth_order(t, u_1, u_2, h):
    u_1_eval = u_2
    u_2_eval = u_2_equation(t, u_1, u_2)

    k_11 = h * u_1_eval
    k_12 = h * u_2_eval
    k_21 = h * (u_2 + k_12 / 2)
    # ! CHANGE IN CASE FUNCTION DEFINITIONS IS DIFFERENT
    k_22 = h * (550 * math.cos(11 * (t + step / 2)) - 5 * (u_2 + k_12 / 2) - 6 * (u_1 + k_11 / 2))
    k_31 = h * (u_2 + k_22 / 2)
    # ! CHANGE IN CASE FUNCTION DEFINITIONS IS DIFFERENT
    k_32 = h * (550 * math.cos(11 * (t + step / 2)) - 5 * (u_2 + k_22 / 2) - 6 * (u_1 + k_21 / 2))
    k_41 = h * (u_2 + k_32)
    # ! CHANGE IN CASE FUNCTION DEFINITIONS IS DIFFERENT
    k_42 = h * (550 * math.cos(11 * (t + step)) - 5 * (u_2 + k_32) - 6 * (u_1 + k_31))

    # ? New values for u_1 and u_2
    new_u_1 = u_1 + (1.0 / 6.0) * (k_11 + 2 * k_21 + 2 * k_31 + k_41)
    new_u_2 = u_2 + (1.0 / 6.0) * (k_12 + 2 * k_22 + 2 * k_32 + k_42)

    row = np.round(
        np.array(
            [
                (u_1, u_2),
                (u_2, u_2_eval),
                (k_11, k_12),
                (k_21, k_22),
                (k_31, k_32),
                (k_41, k_42)
            ]),
        6)

    return new_u_1, new_u_2, row


def generate_kutta_table(start_iteration, iterations, step, u_1, u_2):
    table = []
    q_values = []
    i_values = []

    for itr in range(0, iterations):
        new_values = runge_kutta_fourth_order(
            t=start_iteration,
            u_1=u_1,
            u_2=u_2,
            h=step
        )
        start_iteration += step
        u_1 = new_values[0]
        q_values.append(u_1)
        u_2 = new_values[1]
        i_values.append(u_2)
        table.append(new_values[2])

    return table, q_values, i_values


if __name__ == "__main__":
    print("Metodo de runge kutta 4 orden")
    u_1 = float(input("Ingresa el valor de y(t): "))
    u_2 = float(input("Ingresa el valor de y'(t): "))
    start_iteration = int(input("Ingresa el valor de t: "))
    last_iteration = int(input("Ingresa el valor final de t: "))
    step = float(input("Ingresa el tamanio de paso h: "))

    headers = np.array(["Q(t)/I(t)", "f(u)/f(u')", "k1", "k2", "k3", "k4"])

    # ? Adding one extra to include initial iteration
    iterations = int(abs(last_iteration - start_iteration) / step) + 1
    runge_kutta_table, q_values, i_values = generate_kutta_table(start_iteration, iterations, step, u_1, u_2)
    analytic_tables = []
    analytic_tables.append(analytic_equation_q(start_iteration, iterations, step))
    analytic_tables.append(analytic_equation_i(start_iteration, iterations, step))

    print("TABLA DE RUNGE KUTTA")
    print(tabulate(tabular_data=runge_kutta_table, headers=headers) + "\n")

    print(tabulate(tabular_data=analytic_tables[0], headers=["Q(t)"]) + "\n")
    print(tabulate(tabular_data=analytic_tables[1], headers=["I(t)"]))
    plt.plot(list(np.arange(0, 2.1, 0.1)), analytic_tables[0])
    plt.plot(list(np.arange(0, 2.1, 0.1)), analytic_tables[1])
    plt.plot(list(np.arange(0.1, 2.2, 0.1)), q_values)
    plt.plot(list(np.arange(0.1, 2.2, 0.1)), i_values)
    plt.show()

