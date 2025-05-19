import sys
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QHBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sympy as sp


class FunctionPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculi")

        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-size: 14px;
            }
            QLineEdit, QPushButton {
                padding: 6px;
                border-radius: 6px;
                color : black
            }
            QPushButton {
                background-color: #0055aa;
                color: black;
            }
        """)

        self.layout = QVBoxLayout()

        self.function_input = QLineEdit()
        self.function_input.setPlaceholderText("Enter your function (e.g. x**2, sin(x))....it is fun !")

        limits_layout = QHBoxLayout()
        self.a_input = QLineEdit()
        self.a_input.setPlaceholderText("Lower limit a")
        self.b_input = QLineEdit()
        self.b_input.setPlaceholderText("Upper limit b")
        limits_layout.addWidget(self.a_input)
        limits_layout.addWidget(self.b_input)

        # Taylor series inputs
        taylor_layout = QHBoxLayout()
        self.taylor_a_input = QLineEdit()
        self.taylor_a_input.setPlaceholderText("Taylor expansion point a")
        self.taylor_n_input = QLineEdit()
        self.taylor_n_input.setPlaceholderText("Number of terms n")
        taylor_layout.addWidget(self.taylor_a_input)
        taylor_layout.addWidget(self.taylor_n_input)

        self.plot_button = QPushButton("Plot Function")
        self.plot_button.clicked.connect(self.plot_function)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.layout.addWidget(self.function_input)
        self.layout.addLayout(limits_layout)
        self.layout.addLayout(taylor_layout)
        self.layout.addWidget(self.plot_button)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.error_label)
        self.setLayout(self.layout)

    def plot_function(self):
        self.error_label.setText("")
        func_text = self.function_input.text()
        x_vals = np.linspace(-10, 10, 400)
        x = sp.Symbol('x')

        try:
            func_expr = sp.sympify(func_text, locals={"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log})
            derivative_expr = sp.diff(func_expr, x)
            second_derivative_expr = sp.diff(derivative_expr, x)

            f_numeric = sp.lambdify(x, func_expr, modules=["numpy"])
            f_prime_numeric = sp.lambdify(x, derivative_expr, modules=["numpy"])
            f_double_prime_numeric = sp.lambdify(x, second_derivative_expr, modules=["numpy"])

            y_vals = f_numeric(x_vals)
            y_prime_vals = f_prime_numeric(x_vals)

            crit_points = sp.solve(derivative_expr, x)
            crit_points = [pt.evalf() for pt in crit_points if pt.is_real and -10 <= pt.evalf() <= 10]

            # Limits for area under curve
            try:
                a = float(self.a_input.text())
                b = float(self.b_input.text())
                if a > b:
                    a, b = b, a
                if not (-10 <= a <= 10 and -10 <= b <= 10):
                    raise ValueError("Limits must be between -10 and 10")
            except Exception:
                a, b = None, None

            # Taylor series inputs
            taylor_a = None
            taylor_n = None
            taylor_expr = None
            try:
                if self.taylor_a_input.text() and self.taylor_n_input.text():
                    taylor_a = float(self.taylor_a_input.text())
                    taylor_n = int(self.taylor_n_input.text())
                    series = sp.series(func_expr, x, taylor_a, n=taylor_n)
                    taylor_expr = series.removeO()
                    taylor_func = sp.lambdify(x, taylor_expr, modules=["numpy"])
                    y_taylor_vals = taylor_func(x_vals)
            except Exception as e:
                self.error_label.setText(f"Taylor Series Error: {e}")
                taylor_expr = None

            # Start Plotting
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            ax.plot(x_vals, y_vals, label=f"f(x) = {func_expr}", color="green")
            ax.plot(x_vals, y_prime_vals, label=f"f'(x) = {derivative_expr}", color="red")

            # Plot Taylor series if available
            if taylor_expr is not None:
                ax.plot(x_vals, y_taylor_vals, label=f"Taylor approx at x={taylor_a}, n={taylor_n}", linestyle="--", color="blue")

            # Plot critical points
            for pt in crit_points:
                pt_val = float(pt)
                y_pt = f_numeric(pt_val)
                y_second_derivative = f_double_prime_numeric(pt_val)
                if y_second_derivative > 0:
                    label = "Min"
                    marker = 'o'
                    color = 'green'
                elif y_second_derivative < 0:
                    label = "Max"
                    marker = 's'
                    color = 'purple'
                else:
                    label = "Saddle"
                    marker = 'd'
                    color = 'orange'

                ax.plot(pt_val, y_pt, marker=marker, color=color, markersize=8)
                ax.text(pt_val, y_pt, f" {label}", color=color)

            # Shade area under curve
            if a is not None and b is not None:
                fill_x = np.linspace(a, b, 200)
                fill_y = f_numeric(fill_x)
                ax.fill_between(fill_x, fill_y, color='orange', alpha=0.3,
                                label=f'Area under curve [{a:.2f}, {b:.2f}]')

            ax.legend()
            ax.set_title("Function, Derivative, Critical Points, Area, and Taylor Series")
            self.canvas.draw()

        except Exception as e:
            self.error_label.setText(f"Error: {e}")
            return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FunctionPlotter()
    window.resize(800, 600)
    window.show()
    app.exec()
