import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QWidget,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QCheckBox,
    QInputDialog,
    QToolBar,
    QTextEdit,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PyQt5.QtCore import QSize, Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5.QtGui import QColor
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters

# import scienceplots

import matplotlib.pyplot as plt

import pint

ureg = pint.UnitRegistry()

#plt.style.use("science")

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

# Some convenient absorption energies ...
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
    "Zn-K-beta1": 9572.0,
    "Zn-K-alpha1": 8639.0,
    "Zn-K_alpha2": 8616.0,
}

absorption_energy_list = {}

for key in absorption_energy_list_.keys():
    absorption_energy_list[
        key + ":  " + str(absorption_energy_list_[key]) + " eV"
    ] = absorption_energy_list_[key]

e = 1.60217662e-19
h = 6.62607004e-34
c = 299792458
d = 3.135589312e-10  # d = 3.354e-10


def getMinIndex(x, arr):

    res = np.abs(np.asarray(arr) - np.asarray(x))
    return min(range(len(res)), key=res.__getitem__)


def residual(params, data_points):
    r_solution = params["r"].value
    x0_solution = params["x0"].value

    residuals = []

    # sign = np.sign(
    #     (data_points[0][1] - data_points[1][1])
    #     / (data_points[0][0] - data_points[1][0])
    # )
    sign = 1.0

    for E_data, x_data in data_points:
        E_real = sign * (
            ((1 * ureg.planck_constant * 1 * ureg.speed_of_light))
            / (2 * d * ureg.meter * r_solution)
            * np.sqrt(r_solution**2 + ((x0_solution + x_data) / 2) ** 2)
        ).m_as(ureg.electron_volt)

        residuals.append((E_real - E_data) / 10000.0)

    return np.array(residuals)


def solve_E_x(data_points):

    params = Parameters()
    params.add("r", value=50000, min=1, max=1000000)
    params.add("x0", value=0, min=-500000, max=500000)

    # Perform minimization using Differential Evolution
    result = minimize(
        residual, params, args=(data_points,), method="differential_evolution"
    )

    # Extract optimal values
    r_solution_optimal = result.params["r"].value
    x0_solution_optimal = result.params["x0"].value

    return r_solution_optimal, x0_solution_optimal


# Lorentzian function definition
def lorentzian(x, x0, gamma, A, offset):
    return A * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset


class CalibrationGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("XRTS Calibration Assistant")
        self.setGeometry(200, 200, 1200, 600)
        self.setFixedSize(1200, 600)
        self.setStyleSheet("background-color: #bbbbdd;")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_layout = QHBoxLayout(self.central_widget)
        left_layout = QVBoxLayout()

        label_dropdown_layout = QHBoxLayout()
        self.spectra_label = QLabel("Select File:")
        self.spectra_label.setStyleSheet("font-size: 16px;")
        self.spectra_dropdown = QComboBox()
        self.spectra_dropdown.setFixedSize(QSize(400, 40))
        self.spectra_dropdown.currentIndexChanged.connect(
            self.on_spectrum_selected
        )
        label_dropdown_layout.addWidget(self.spectra_label)
        label_dropdown_layout.addWidget(self.spectra_dropdown)
        left_layout.addLayout(label_dropdown_layout)

        self.load_button = QPushButton("Load Spectra")
        self.load_button.setStyleSheet(
            "font-size: 16px; font-weight: bold; background-color: #99aabb"
        )
        self.load_button.setFixedSize(QSize(200, 40))
        self.load_button.clicked.connect(self.load_spectra)
        left_layout.addWidget(self.load_button)

        # Quadratic fit checkbox
        self.linear_fit = True  # Default is linear
        self.quadratic_checkbox = QCheckBox("Quadratic Fit")
        self.quadratic_checkbox.setStyleSheet("font-size: 16px;")
        self.quadratic_checkbox.stateChanged.connect(
            lambda state: setattr(self, "linear_fit", state != Qt.Checked)
        )
        left_layout.addWidget(self.quadratic_checkbox)

        calibrate_and_console_layout = QHBoxLayout()
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.setStyleSheet(
            "font-size: 16px; font-weight: bold; background-color: #5577aa;"
        )
        self.calibrate_button.setFixedSize(QSize(200, 80))
        self.calibrate_button.clicked.connect(self.calibrate)
        calibrate_and_console_layout.addWidget(self.calibrate_button)

        self.console = QTextEdit()
        self.console.setFixedSize(QSize(295, 200))
        self.console.setReadOnly(True)
        calibrate_and_console_layout.addWidget(self.console)

        left_layout.addLayout(calibrate_and_console_layout)
        left_layout.addStretch(1)

        self.peak_list_label = QLabel("Selected peaks:")
        self.peak_list_label.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )
        left_layout.addWidget(self.peak_list_label)

        self.peak_list = QListWidget()
        self.peak_list.setSpacing(-5)
        self.peak_list.setStyleSheet("background-color: lightgrey;")
        self.peak_list.setFixedSize(500, 250)
        left_layout.addWidget(self.peak_list)

        main_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.set_facecolor("whitesmoke")
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        main_layout.addLayout(right_layout)

        self.spectra_files = []
        self.spectra_data = []
        self.selected_peaks = []
        self.peak_labels = []
        self.current_spectrum_index = 0
        self.state = {}

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Pixel position", fontsize=16)
        self.ax.set_ylabel("Intensity [arb. units]", fontsize=16)
        self.ax.grid(True)

        self.textToConsole(r"<b>How does this GUI work?<b>")
        self.textToConsole(
            '1. Select the folder with your calibration spectra (.csv or .txt) by clicking on the "Load Spectra" button on top.'
        )
        self.textToConsole(
            '2. For each of the spectra you want to use, first select the spectra file on the top and then select the corresponding peak in the graph on the right by drawing the region of interest. You may delete peaks again by clicking "Delete".'
        )
        self.textToConsole(
            '3. Select for each peak the corresponding energy in the dropdown menu right next to the entry in the "Selected peaks" widget.'
        )
        self.textToConsole(
            '4. Once you selected all peaks you want to utilize for the calibration, click on "Calibrate".'
        )
        self.textToConsole(
            "5. The calibration (consisting of px position - energy pairs) is saved in the folder selected in step 1 with the name prompted by the user."
        )

    def textToConsole(self, text):
        """Append text to the console."""
        self.console.append(text)

    def load_spectra(self):
        # Open folder dialog to select spectra folder

        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Spectra Folder"
        )

        self.folder_path = folder_path

        if folder_path:

            self.selected_peaks = (
                []
            )  # To store (file, peak_index, line_object)
            self.peak_labels = []
            self.current_spectrum_index = 0
            self.state = (
                {}
            )  # Dictionary to hold the state for each spectrum file
            self.spectra_files.clear()
            self.spectra_data.clear()
            self.spectra_dropdown.clear()  # Clear any previous dropdown items

            for file in os.listdir(folder_path):

                if "dispersion" not in file:
                    if file.endswith(".txt") or file.endswith(
                        ".csv"
                    ):  # Assuming spectra are .txt files, change as needed
                        file_path = os.path.join(folder_path, file)
                        try:
                            try:
                                dataPx, dataI = np.genfromtxt(
                                    file_path,
                                    delimiter=",",
                                    unpack=True,
                                    skip_header=1,
                                )  # Load the spectra data
                            except:
                                dataPx, dataI = np.genfromtxt(
                                    file_path, unpack=True, skip_header=1
                                )  # Load the spectra data
                        except:
                            # Do nothing if no file is found.
                            return -1
                        self.spectra_files.append(file)
                        self.spectra_data.append([dataPx, dataI])
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
            color="C0",
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
        spectrum_x = self.spectra_data[self.current_spectrum_index][0]

        lower_index = getMinIndex(int(xmin), spectrum_x)
        upper_index = getMinIndex(int(xmax), spectrum_x)

        try:
            peak_index = self.fit_lorentzian(lower_index, upper_index)
        except:
            return -1

        if peak_index is not None:
            # Draw vertical line for the selected peak
            vertical_line = self.ax.axvline(
                peak_index,
                color="black",
                linestyle="--",
                ymin=0,
                ymax=1,
                linewidth=2,
            )
            self.canvas.draw()

            spectrum_file = self.spectra_files[self.current_spectrum_index]
            peak_entry = (
                spectrum_file,
                peak_index,
                vertical_line,
            )  
            self.selected_peaks.append(peak_entry)

            # Add the peak to the list
            self.add_peak_to_list(peak_index)

            # Update the state with the new peak
            self.update_state_with_new_peak(peak_index)

    def update_dropdown_appearance(self):
        """Update the appearance of dropdown items based on whether they have peaks"""
        for i in range(self.spectra_dropdown.count()):
            file_name = self.spectra_dropdown.itemText(i)
            if (
                file_name in self.state
                and len(self.state[file_name]["peaks"]) > 0
            ):
                # File has peaks - set a colored background
                self.spectra_dropdown.setItemData(
                    i, QColor("royalblue"), Qt.BackgroundRole
                )
            else:
                # File has no peaks - clear any background color
                self.spectra_dropdown.setItemData(
                    i, QColor("#e6e6f5"), Qt.BackgroundRole
                )

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
        self.update_dropdown_appearance()

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
            if np.min(x_data) <= fitted_center <= np.max(x_data):

                return fitted_center  # Return the integer index of the fitted peak center
            else:
                print("Fit not successfull.")
                raise ValueError  # Return None if the center is outside the selected range
        except (RuntimeError, ValueError):
            # Return None if the fit fails
            res = (
                x_data[lower_index]
                + x_data[
                    np.nanargmax(
                        np.array(spectrum[1][lower_index:upper_index])
                    )
                ]
            )

            return res

    def add_peak_to_list(self, index):
        # Create a QWidget to hold the peak information and dropdown
        peak_item_widget = QWidget()

        # Layout for the peak information and dropdown
        layout = QHBoxLayout()

        energy_value = list(absorption_energy_list.keys())[0]
        # Label to display peak range and absorption
        peak_label = QLabel(f"Peak Position [px]: {index:.2f} ")
        layout.addWidget(peak_label)

        dropdown = QComboBox()
        dropdown.addItems(absorption_energy_list.keys())
        dropdown.addItem("Custom")
        dropdown.setFixedSize(QSize(200, 25))  # Adjust size as needed
        layout.addWidget(dropdown)

        # Connect dropdown change to the update method
        dropdown.currentTextChanged.connect(
            lambda: self.update_energy_state(
                index, dropdown.currentText(), dropdown_widget=dropdown
            )
        )

        # Button to delete the peak
        delete_button = QPushButton("Delete")
        delete_button.setFixedSize(QSize(60, 25))  # Adjust size as needed
        delete_button.clicked.connect(
            lambda: self.delete_peak(peak_item_widget, index)
        )
        layout.addWidget(delete_button)

        # Set the layout and add it to the QListWidget
        peak_item_widget.setLayout(layout)
        list_item = QListWidgetItem()
        list_item.setSizeHint(
            peak_item_widget.sizeHint()
        )  # Ensure proper size in QListWidget
        self.peak_list.addItem(list_item)
        self.peak_list.setItemWidget(list_item, peak_item_widget)
        self.update_dropdown_appearance()

    def update_energy_state(
        self, peak_index, selected_energy, dropdown_widget=None
    ):
        current_file = self.spectra_files[self.current_spectrum_index]

        # If user selects "Custom", prompt for input
        if selected_energy == "Custom":
            custom_value, ok = QInputDialog.getDouble(
                self,
                "Custom Energy",
                f"Enter energy value in eV:",
                decimals=2,
            )
            if not ok:
                return  # User cancelled input

            selected_energy = f"Custom: {custom_value:.2f} eV"

            absorption_energy_list[f"Custom: {custom_value:.2f} eV"] = float(
                f"{custom_value:.2f}"
            )

            # Update the dropdown UI if a widget was passed
            if dropdown_widget is not None:
                index = dropdown_widget.findText("Custom")
                if index != -1:
                    dropdown_widget.setItemText(
                        index, f"Custom: {custom_value:.2f} eV"
                    )
                else:
                    dropdown_widget.addItem(selected_energy)
                    dropdown_widget.setCurrentText(
                        f"Custom: {custom_value:.2f} eV"
                    )
                self.canvas.draw()

        # Update the state for the current file
        if current_file in self.state:
            peaks = self.state[current_file]["peaks"]
            dropdowns = self.state[current_file]["dropdowns"]

            if peak_index in peaks:
                index = peaks.index(peak_index)
                if index < len(dropdowns):
                    dropdowns[index] = selected_energy
                else:
                    dropdowns.append(selected_energy)

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
                        peak[
                            2
                        ].remove()  # Remove the vertical line from the plot
                        self.selected_peaks.remove(peak)
                        self.canvas.draw()  # Redraw the canvas without the removed line

                        # Also remove the peak from the state
                        self.remove_peak_from_state(peak_index)
                        break
                break
        self.update_dropdown_appearance()

    def remove_peak_from_state(self, peak_index):
        current_file = self.spectra_files[self.current_spectrum_index]

        if current_file in self.state:
            if peak_index in self.state[current_file]["peaks"]:
                index = self.state[current_file]["peaks"].index(peak_index)
                # Remove the peak and the associated energy
                self.state[current_file]["peaks"].pop(index)
                self.state[current_file]["dropdowns"].pop(index)
        self.update_dropdown_appearance()

    def save_current_state(self):
        try:
            current_file = self.spectra_files[self.current_spectrum_index]
        except:
            return -1
        if current_file not in self.state:
            self.state[current_file] = {"peaks": [], "dropdowns": []}

        # Save peak positions (x) and associated vertical line information
        self.state[current_file]["peaks"] = [
            peak[1] for peak in self.selected_peaks
        ]

        # Save dropdown values
        dropdown_values = []
        for i in range(self.peak_list.count()):
            item = self.peak_list.item(i)
            widget = self.peak_list.itemWidget(item)
            dropdown = (
                widget.layout().itemAt(1).widget()
            )  # Retrieve the dropdown
            dropdown_values.append(dropdown.currentText())
        self.state[current_file]["dropdowns"] = dropdown_values

    def restore_state(self):
        try:
            current_file = self.spectra_files[self.current_spectrum_index]
        except:
            return -1

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
                dropdown = (
                    widget.layout().itemAt(1).widget()
                )  # Retrieve the dropdown
                dropdown.setCurrentText(dropdowns[i])

    def calibrate(self):

        if len(self.state.items()) == 0:
            print("Please select peaks, before calibrating.")
            return -1

        filename, ok = QInputDialog.getText(
            self, "Save Calibration", "Enter filename (without extension):"
        )

        if not ok or not filename.strip():
            filename = "dispersion"
        else:
            filename = filename.strip()

        positions = []
        energies = []

        # Iterate through each file in the saved state
        for spectrum_file, state_data in self.state.items():
            peaks = state_data["peaks"]
            dropdowns = state_data["dropdowns"]

            # print(f"\nFile: {spectrum_file}")

            if len(peaks) != len(dropdowns):
                print(
                    "Warning: Number of peaks and dropdown selections do not match."
                )

            # Print peak positions and selected energies
            for i in range(len(peaks)):

                positions.append(peaks[i])
                energies.append(dropdowns[i])

                peak_index = peaks[i]
                selected_energy = dropdowns[i] if i < len(dropdowns) else "N/A"

        if len(positions) <= 1:
            button = QMessageBox.warning(
                self,
                "Warning",
                "At least 2 peaks needed for calibration!",
                buttons=QMessageBox.Ok,
            )
            return -1

        if len(set(energies)) == 1:
            button = QMessageBox.warning(
                self,
                "Warning",
                "Please select at least 2 different peak energies!",
                buttons=QMessageBox.Ok,
            )
            return -1

        data_points = [
            [absorption_energy_list[e], pos]
            for e, pos in zip(energies, positions)
        ]

        r_values = []
        x0_values = []

        # Solve for r and x0 for the group of data points
        r, x0 = solve_E_x(data_points)

        x = self.spectra_data[self.current_spectrum_index][0]
        # Calculate corresponding E values using the average parameters
        E_real = (
            ((ureg.planck_constant * ureg.speed_of_light))
            / (2 * d * ureg.meter * r)
            * np.sqrt(r**2 + ((x0 + x) / (2)) ** 2)
        ).m_as(ureg.electron_volt)

        x_data = np.array([data[1] for data in data_points])
        E_data = np.array([data[0] for data in data_points])

        if self.linear_fit:
            fit = np.polyfit(x_data, E_data, 1)
            slope, intercept = fit

            E_linear = np.polyval(fit, x)

            np.savetxt(
                self.folder_path + f"/{filename}.txt",
                np.column_stack((x, E_linear)),
                delimiter=",",
                header="Pixel position, Energy",
            )

            print(
                'Calibration saved under "'
                + self.folder_path
                + f'/{filename}.txt"'
            )

            # dE_real = np.diff(E_real)
            # dE_linear = np.diff(E_linear)
            deviation = E_real - E_linear
            fig, axs = plt.subplots(ncols=2, figsize=(13, 5))

            axs[0].plot(
                x,
                E_linear,
                label=f"Linear fit: $E[ev](x) = {slope:.4f}x + {intercept:.2f}$",
                linestyle="--",
                color="g",
            )
        else:
            fit = np.polyfit(x_data, E_data, 2)
            a, b, c = fit

            E_quadratic = np.polyval(fit, x)

            np.savetxt(
                self.folder_path + f"/{filename}.txt",
                np.column_stack((x, E_quadratic)),
                delimiter=",",
                header="Pixel position, Energy",
            )

            print(
                'Calibration saved under "'
                + self.folder_path
                + f'/{filename}.txt"'
            )

            # dE_real = np.diff(E_real)
            # dE_linear = np.diff(E_linear)
            deviation = E_real - E_quadratic
            fig, axs = plt.subplots(ncols=2, figsize=(13, 5))

            axs[0].plot(
                x,
                E_quadratic,
                label=f"Quadratic fit: E[eV](x) = ${a:.7f}x^2 + {b:.2f}x + {c:.2f}$",
                linestyle="--",
                color="g",
            )

        axs[0].plot(
            x,
            E_real,
            label=r"Van Hamos Dispersion: $E(x) = \frac{hc}{2rd} \sqrt{r^2 + \left(\frac{x_0 + x}{2}\right)^2}$",
        )

        axs[0].scatter(
            x_data, E_data, marker="x", color="r", s=20, label=r"Data"
        )
        axs[0].set_xlabel(r"Pixel position")
        axs[0].set_ylabel("Energy [eV]")

        axs[0].grid(True)

        count = 0

        xlims = [np.inf, -np.inf]

        for i, sp in enumerate(self.spectra_data):
            xv, yv = sp
            for l, p in enumerate(
                self.state[list(self.state.keys())[i]]["peaks"]
            ):
                axs[1].axvline(
                    np.polyval(fit, p),
                    label=self.state[list(self.state.keys())[i]]["dropdowns"][
                        l
                    ],
                    color=colors_peaks[count],
                    linestyle="--",
                    ymin=0,
                    ymax=1,
                )
                count += 1

            if len(self.state[list(self.state.keys())[i]]["peaks"]) > 0:
                axs[1].plot(
                    np.polyval(fit, xv), yv / np.nanmax(yv), color=f"C{i}"
                )
                axs[1].axvline(ymin=0.0, ymax=1.0)

                if xlims[0] > np.nanmin(np.polyval(fit, xv)):
                    xlims[0] = np.nanmin(np.polyval(fit, xv))

                if xlims[1] < np.nanmax(np.polyval(fit, xv)):
                    xlims[1] = np.nanmax(np.polyval(fit, xv))

        axs[1].set_xlim(xlims[0], xlims[1])
        axs[1].grid(True)

        axins = inset_axes(
            axs[0], width="35%", height="25%", loc="lower left", borderpad=4
        )
        axins.plot(x, deviation, color="orange", label="Deviation")
        axins.set_title("Deviation", fontsize=10)
        axins.set_xlabel("Pixel", fontsize=8)
        axins.set_ylabel(r"$\Lambda$E [eV]", fontsize=8)
        axins.tick_params(axis="both", which="major", labelsize=5)
        axins.grid(True)

        axs[1].set_xlabel(r"Energy [eV]")
        axs[1].set_ylabel("Intensity [arb. units]")
        axs[0].set_title("Dispersion")
        axs[1].set_title("Calibrated spectra")
        axs[0].legend(fontsize=9, loc="upper right")
        fig.canvas.manager.set_window_title("Calibration result")
        axs[1].legend(fontsize=12)
        plt.tight_layout()
        fig.savefig(self.folder_path + "/dispersion_overview.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CalibrationGUI()
    gui.show()
    sys.exit(app.exec_())
