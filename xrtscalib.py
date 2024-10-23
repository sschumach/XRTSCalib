import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QWidget,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QToolBar,
    QTextEdit,
)
from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit, minimize
import scienceplots

import matplotlib.pyplot as plt

import pint

ureg = pint.UnitRegistry()

plt.style.use("science")

colors_peaks = [
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "gold",
    "lime",
    "navy",
    "teal",
    "coral",
    "aqua",
    "indigo",
    "violet",
]

absorption_energy_list_ = {
    "Co-K-alpha2": 6915.538,
    "Co-K-alpha1": 6930.378,
    "Co-K-beta1": 7649.4,
    "Cu-K-alpha1": 8048.11,
    "Cu-K-alpha2": 8028.38,
    "Ni-K-alpha2": 7461.034,
    "Ni-K-alpha1": 7478.252,
    "Ni-K-beta1": 8264.77,
    "Cu-K-alpha2": 8027.842,
    "Cu-K-alpha1": 8047.823,
    "Cu-K-beta1": 8905.29,
    "LCLS": 8165.0,
}

absorption_energy_list = {}

for key in absorption_energy_list_.keys():
    absorption_energy_list[key + ":  " + str(absorption_energy_list_[key]) + " eV"] = (
        absorption_energy_list_[key]
    )

e = 1.60217662e-19
h = 6.62607004e-34
c = 299792458
d = 3.135589312e-10  # d = 3.354e-10

def residual(params, data_points):
    r_solution, x0_solution = params
    error = 0.0
    for E_data, x_data in data_points:
        E_real = (((ureg.planck_constant * ureg.speed_of_light)) / (2 * d * ureg.meter) * np.sqrt(
            1 + ((x0_solution + x_data) / (2*r_solution)) ** 2)).m_as(ureg.electron_volt)
        error += (E_real - E_data) ** 2

    # print(r_solution, x0_solution, error)
    return error


def solve_E_x(data_points):

    initial_guess = [1000, 5000.0]  # Initial guesses for r_solution and x0_solution
    result = minimize(
        residual, initial_guess, args=(data_points), method='SLSQP', options={'xtol': 1e-20, 'disp': False}
    )  # Minimize the objective function
    r_solution_optimal, x0_solution_optimal = result.x  # Extract optimal parameters
    return r_solution_optimal, x0_solution_optimal


# Lorentzian function definition
def lorentzian(x, x0, gamma, A, offset):
    return A * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset


class CalibrationGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Calibration Helper")
        self.setGeometry(200, 200, 1200, 600)  # Adjust width to fit peak list
        self.setFixedSize(1200, 600)
        # self.setStyleSheet("background-color: lavender;")
        # Central widget to hold everything
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout (horizontal): combo box and buttons on the left, plot on the right
        main_layout = QHBoxLayout(self.central_widget)

        # Left layout for the dropdown, buttons, and peak list
        left_layout = QVBoxLayout()

        # Create a horizontal layout for the label and spectra dropdown
        label_dropdown_layout = QHBoxLayout()

        # Create the label for the spectra dropdown
        self.spectra_label = QLabel("Select File:")
        self.spectra_label.setStyleSheet("font-size: 16px;")
        # Dropdown to select the spectrum
        self.spectra_dropdown = QComboBox()
        self.spectra_dropdown.setFixedSize(QSize(400, 40))
        self.spectra_dropdown.currentIndexChanged.connect(self.on_spectrum_selected)

        # Add the label and dropdown to the label_dropdown_layout
        label_dropdown_layout.addWidget(self.spectra_label)
        label_dropdown_layout.addWidget(self.spectra_dropdown)

        # Add the label and dropdown layout to the left_layout
        left_layout.addLayout(label_dropdown_layout)

        # Button to load spectra (smaller button)
        self.load_button = QPushButton("Load Spectra")
        self.load_button.setStyleSheet("font-size: 16px;")
        self.load_button.setFixedSize(QSize(200, 40))  # Set smaller button size
        self.load_button.clicked.connect(self.load_spectra)
        left_layout.addWidget(self.load_button)

        # Create a horizontal layout for the calibrate button and console side by side
        calibrate_and_console_layout = QHBoxLayout()

        # Button to calibrate (smaller button)
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.setStyleSheet("font-size: 16px; background-color: firebrick;")
        self.calibrate_button.setFixedSize(QSize(200, 80))  # Set smaller button size
        # self.calibrate_button.setStyleSheet("background-color: red;")
        self.calibrate_button.clicked.connect(self.calibrate)
        calibrate_and_console_layout.addWidget(self.calibrate_button)

        # Console (text output area) to show output
        self.console = QTextEdit()
        self.console.setFixedSize(QSize(295, 200))  # Adjust size as needed
        self.console.setReadOnly(True)  # Make it read-only
        calibrate_and_console_layout.addWidget(self.console)

        # Add the calibrate_and_console_layout to the left_layout
        left_layout.addLayout(calibrate_and_console_layout)

        # Add spacing between buttons (optional for better spacing)
        left_layout.addStretch(1)
        self.peak_list_label = QLabel("Selected peaks:")
        self.peak_list_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(self.peak_list_label)  # Add the label above the peak list
        # List to display selected peaks
        self.peak_list = QListWidget()
        self.peak_list.setSpacing(-5)
        self.peak_list.setStyleSheet("background-color: linen;")
        self.peak_list.setFixedSize(500, 250)  # Adjust width to make the list wider
        left_layout.addWidget(self.peak_list)

        # Add the left layout to the main layout
        main_layout.addLayout(left_layout)

        # Right layout for the plot and toolbar
        right_layout = QVBoxLayout()

        # Matplotlib figure for plotting spectra (on the right)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.set_facecolor("whitesmoke")

        # Create a toolbar for Matplotlib navigation (zoom/pan)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add the toolbar on top of the canvas
        right_layout.addWidget(self.toolbar)  # Add the toolbar above the canvas
        right_layout.addWidget(self.canvas)

        main_layout.addLayout(right_layout)

        # Variables to hold the spectra and selected regions
        self.spectra_files = []
        self.spectra_data = []
        self.selected_peaks = []  # To store (file, peak_index, line_object)
        self.peak_labels = []
        self.current_spectrum_index = 0
        self.state = {}  # Dictionary to hold the state for each spectrum file

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Pixel position", fontsize=16)
        self.ax.set_ylabel("Intensity [arb. units]", fontsize=16)
        self.ax.grid(True)

    def textToConsole(self, text):
        """Append text to the console."""
        self.console.append(text)

    def load_spectra(self):
        # Open folder dialog to select spectra folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Spectra Folder")

        if folder_path:
            self.spectra_files.clear()
            self.spectra_data.clear()
            self.spectra_dropdown.clear()  # Clear any previous dropdown items

            for file in os.listdir(folder_path):
                if file.endswith(".txt") or file.endswith(
                    ".csv"
                ):  # Assuming spectra are .txt files, change as needed
                    file_path = os.path.join(folder_path, file)
                    dataPx, dataI = np.genfromtxt(
                        file_path, delimiter=",", unpack=True, skip_header=1
                    )  # Load the spectra data
                    self.spectra_files.append(file)
                    self.spectra_data.append([range(0, len(dataI)), dataI])
                    self.state[file] = {
                        "peaks": [],
                        "dropdowns": [],
                    }  # Initialize state for each file

            # Populate the dropdown with the file names
            if self.spectra_files:
                self.spectra_dropdown.addItems(self.spectra_files)
                self.plot_spectrum(0)  # Plot the first spectrum by default

    def on_spectrum_selected(self, index):
        self.save_current_state()  # Save current state before switching
        self.current_spectrum_index = index
        self.restore_state()  # Restore state for the new spectrum

    def plot_spectrum(self, index):

        spectrum = self.spectra_data[index]
        self.ax.plot(
            spectrum[0],
            spectrum[1] / np.nanmax(spectrum[1]),
            label=f"Spectrum {self.spectra_files[index]}",
            linewidth=1,
        )

        # Add SpanSelector for peak selection
        self.span = SpanSelector(
            self.ax,
            self.onselect,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.6, facecolor="royalblue"),
        )

        self.canvas.draw()  # Redraw the canvas with the new plot

    def onselect(self, xmin, xmax):
        # Fit the selected region to a Lorentzian
        lower_index = int(xmin)
        upper_index = int(xmax)
        peak_index = self.fit_lorentzian(lower_index, upper_index)

        if peak_index is not None:
            # Draw vertical line for the selected peak
            vertical_line = self.ax.axvline(
                peak_index, color="black", linestyle="--", ymin=0, ymax=1, linewidth=2
            )
            self.canvas.draw()

            # Add this selection to the list
            spectrum_file = self.spectra_files[self.current_spectrum_index]
            peak_entry = (
                spectrum_file,
                peak_index,
                vertical_line,
            )  # Store vertical line object
            self.selected_peaks.append(peak_entry)

            # Add the peak to the QListWidget
            self.add_peak_to_list(peak_index)

            # Update the state with the new peak
            self.update_state_with_new_peak(peak_index)

    def update_state_with_new_peak(self, peak_index):
        current_file = self.spectra_files[self.current_spectrum_index]

        # Ensure the state for the current file exists
        if current_file not in self.state:
            self.state[current_file] = {"peaks": [], "dropdowns": []}

        # Append the new peak index and a default energy to the state's peaks and dropdowns
        self.state[current_file]["peaks"].append(peak_index)
        self.state[current_file]["dropdowns"].append(
            list(absorption_energy_list.keys())[0]
        )  # Default selection

    def fit_lorentzian(self, lower_index, upper_index):
        # Get the spectrum data in the selected range
        spectrum = self.spectra_data[self.current_spectrum_index]
        x_data = np.array(spectrum[0][lower_index:upper_index])
        y_data = np.array(spectrum[1][lower_index:upper_index])

        # Initial guess for Lorentzian fit parameters: (x0, gamma, A, offset)
        initial_guess = [
            np.mean(x_data),
            (upper_index - lower_index) / 2,
            np.max(y_data),
            0.0,
        ]

        try:
            # Fit the Lorentzian curve to the selected data
            popt, _ = curve_fit(lorentzian, x_data, y_data, p0=initial_guess)

            # Extract the fitted center (x0) from the fit
            fitted_center = popt[0]

            # Check if the fitted center is within the selected region
            if lower_index <= fitted_center <= upper_index:
                return (
                    fitted_center  # Return the integer index of the fitted peak center
                )
            else:
                raise ValueError  # Return None if the center is outside the selected range
        except (RuntimeError, ValueError):
            # Return None if the fit fails
            return lower_index + np.nanargmax(
                np.array(spectrum[1][lower_index:upper_index])
            )

    def add_peak_to_list(self, index):
        # Create a QWidget to hold the peak information and dropdown
        peak_item_widget = QWidget()

        # Layout for the peak information and dropdown
        layout = QHBoxLayout()

        energy_value = list(absorption_energy_list.keys())[0]
        # Label to display peak range and absorption
        peak_label = QLabel(f"Peak Position [px]: {index:.2f} ")
        layout.addWidget(peak_label)

        # Dropdown (QComboBox) for selecting a value between 0 and 10
        dropdown = QComboBox()
        dropdown.addItems(absorption_energy_list.keys())
        dropdown.setFixedSize(QSize(200, 25))  # Adjust size as needed
        layout.addWidget(dropdown)

        # Connect dropdown change to the update method
        dropdown.currentTextChanged.connect(
            lambda: self.update_energy_state(index, dropdown.currentText())
        )

        # Button to delete the peak
        delete_button = QPushButton("Delete")
        delete_button.setFixedSize(QSize(60, 25))  # Adjust size as needed
        delete_button.clicked.connect(lambda: self.delete_peak(peak_item_widget, index))
        layout.addWidget(delete_button)

        # Set the layout and add it to the QListWidget
        peak_item_widget.setLayout(layout)
        list_item = QListWidgetItem()
        list_item.setSizeHint(
            peak_item_widget.sizeHint()
        )  # Ensure proper size in QListWidget
        self.peak_list.addItem(list_item)
        self.peak_list.setItemWidget(list_item, peak_item_widget)

    def update_energy_state(self, peak_index, selected_energy):
        current_file = self.spectra_files[self.current_spectrum_index]

        # Update the state for the current file
        if current_file in self.state:
            peaks = self.state[current_file]["peaks"]
            dropdowns = self.state[current_file]["dropdowns"]

            # Find the index of the peak to update
            if peak_index in peaks:
                index = peaks.index(peak_index)
                # Update the selected energy in the dropdowns
                if index < len(dropdowns):
                    dropdowns[index] = selected_energy
                else:
                    dropdowns.append(
                        selected_energy
                    )  # In case of new peaks added after saving state

    def delete_peak(self, peak_widget, peak_index):
        # Find the index of the peak widget to remove
        for i in range(self.peak_list.count()):
            item = self.peak_list.item(i)
            widget = self.peak_list.itemWidget(item)
            if widget == peak_widget:
                self.peak_list.takeItem(i)
                # Remove the corresponding peak data and vertical line
                for peak in self.selected_peaks:
                    if peak[1] == peak_index:
                        peak[2].remove()  # Remove the vertical line from the plot
                        self.selected_peaks.remove(peak)
                        self.canvas.draw()  # Redraw the canvas without the removed line

                        # Also remove the peak from the state
                        self.remove_peak_from_state(peak_index)
                        break
                break

    def remove_peak_from_state(self, peak_index):
        current_file = self.spectra_files[self.current_spectrum_index]

        if current_file in self.state:
            if peak_index in self.state[current_file]["peaks"]:
                index = self.state[current_file]["peaks"].index(peak_index)
                # Remove the peak and the associated energy
                self.state[current_file]["peaks"].pop(index)
                self.state[current_file]["dropdowns"].pop(index)

    def save_current_state(self):
        current_file = self.spectra_files[self.current_spectrum_index]
        if current_file not in self.state:
            self.state[current_file] = {"peaks": [], "dropdowns": []}

        # Save peak positions (x) and associated vertical line information
        self.state[current_file]["peaks"] = [peak[1] for peak in self.selected_peaks]

        # Save dropdown values
        dropdown_values = []
        for i in range(self.peak_list.count()):
            item = self.peak_list.item(i)
            widget = self.peak_list.itemWidget(item)
            dropdown = widget.layout().itemAt(1).widget()  # Retrieve the dropdown
            dropdown_values.append(dropdown.currentText())
        self.state[current_file]["dropdowns"] = dropdown_values

    def restore_state(self):
        current_file = self.spectra_files[self.current_spectrum_index]

        # Clear existing selections and plot
        self.selected_peaks.clear()
        self.peak_list.clear()
        self.ax.clear()
        self.ax.set_xlabel("Pixel position", fontsize=16)
        self.ax.set_ylabel("Intensity [arb. units]", fontsize=16)
        self.ax.grid(True)
        # Re-plot the spectrum data
        self.plot_spectrum(self.current_spectrum_index)

        # Restore peaks and vertical lines if the file has saved state
        if current_file in self.state:
            for peak_index in self.state[current_file]["peaks"]:
                # Redraw the vertical line on the plot
                vertical_line = self.ax.axvline(
                    peak_index,
                    color="black",
                    linestyle="--",
                    ymin=0,
                    ymax=1,
                    linewidth=2,
                )
                self.selected_peaks.append(
                    (current_file, peak_index, vertical_line)
                )  # Save the vertical line info
                self.add_peak_to_list(peak_index)
                self.canvas.draw()
            # Restore dropdown selections
            dropdowns = self.state[current_file]["dropdowns"]
            for i in range(self.peak_list.count()):
                item = self.peak_list.item(i)
                widget = self.peak_list.itemWidget(item)
                dropdown = widget.layout().itemAt(1).widget()  # Retrieve the dropdown
                dropdown.setCurrentText(dropdowns[i])

    def calibrate(self):

        if len(self.state.items()) == 0:
            print("Please select peaks, before calibrating.")
            return -1

        # print("#######################")
        # print("Calibration Results:")
        positions = []
        energies = []

        # Iterate through each file in the saved state
        for spectrum_file, state_data in self.state.items():
            peaks = state_data["peaks"]
            dropdowns = state_data["dropdowns"]

            # print(f"\nFile: {spectrum_file}")

            if len(peaks) != len(dropdowns):
                print("Warning: Number of peaks and dropdown selections do not match.")

            # Print peak positions and selected energies
            for i in range(len(peaks)):

                positions.append(peaks[i])
                energies.append(dropdowns[i])

                peak_index = peaks[i]
                selected_energy = dropdowns[i] if i < len(dropdowns) else "N/A"
                # print(
                #     f"Peak Position [x]: {peak_index:.2f}, Selected Energy: {selected_energy}"
                # )

        # print(positions, energies)
        # print("#######################")

        data_points = [
            [absorption_energy_list[e], pos] for e, pos in zip(energies, positions)
        ]

        r_values = []
        x0_values = []

        # Solve for r and x0 for the group of data points
        r, x0 = solve_E_x(data_points)

        # Calculate the average values of r and x0
        # average_r = np.mean(r_values)
        # average_x0 = np.mean(x0_values)
        x = range(len(self.spectra_data[0][1]))

        # Calculate corresponding E values using the average parameters
        E_real = (((ureg.planck_constant * ureg.speed_of_light)) / (2 * d * ureg.meter) * np.sqrt(
            1 + ((x0 + x) / (2*r)) ** 2)).m_as(ureg.electron_volt)

        # Linear fit for the three data points
        x_data = np.array([data[1] for data in data_points])
        E_data = np.array([data[0] for data in data_points])

        linear_fit = np.polyfit(x_data, E_data, 1)
        slope, intercept = linear_fit

        E_linear = np.polyval(linear_fit, x)
        # dE_real = np.diff(E_real)
        # dE_linear = np.diff(E_linear)

        fig, axs = plt.subplots(ncols=2, figsize=(13, 5))

        axs[0].plot(
            x,
            E_linear,
            label=f"Linear ~ fit: $E(x) = {slope:.4f}x + {intercept:.2f}$",
            linestyle="--",
            color="g",
        )
        axs[0].plot(
            x,
            E_real,
            label=r"Real: $E(x) = \frac{hc}{2rd} \sqrt{r^2 + \left(\frac{x_0 + x}{2}\right)^2}$",
        )
        axs[0].scatter(x_data, E_data, marker="x", color="r", s=20, label=r"Data")
        axs[0].set_xlabel(r"Pixel position")
        axs[0].set_ylabel("Energy [eV]")

        axs[0].grid(True)

        count = 0

        xlims = [np.inf, -np.inf]

        for i, sp in enumerate(self.spectra_data):
            xv, yv = sp
            for l, p in enumerate(self.state[list(self.state.keys())[i]]["peaks"]):
                axs[1].axvline(
                    np.polyval(linear_fit, p),
                    label=self.state[list(self.state.keys())[i]]["dropdowns"][l],
                    color=colors_peaks[count],
                    linestyle="--",
                    ymin=0,
                    ymax=1,
                )
                count += 1

            if len(self.state[list(self.state.keys())[i]]["peaks"]) > 0:
                axs[1].plot(
                    np.polyval(linear_fit, xv), yv / np.nanmax(yv), color=f"C{i}"
                )
                axs[1].axvline(ymin=0.0, ymax=1.0)

                if xlims[0] > np.nanmin(np.polyval(linear_fit, xv)):
                    xlims[0] = np.nanmin(np.polyval(linear_fit, xv))

                if xlims[1] < np.nanmax(np.polyval(linear_fit, xv)):
                    xlims[1] = np.nanmax(np.polyval(linear_fit, xv))

        axs[1].set_xlim(xlims[0], xlims[1])
        axs[1].grid(True)

        axs[1].set_xlabel(r"Energy [eV]")
        axs[1].set_ylabel("Intensity [arb. units]")
        axs[0].set_title("Dispersion")
        axs[1].set_title("Calibrated spectra")
        axs[0].legend(fontsize=13)
        # fig.canvas.set_window_title('Calibration result')
        axs[1].legend(fontsize=12, bbox_to_anchor=(1.02, 1.05))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CalibrationGUI()
    gui.show()
    sys.exit(app.exec_())
