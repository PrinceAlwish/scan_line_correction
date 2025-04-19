"""
Landsat 7 SLC-off Scan Line Correction for QGIS
==============================================

This script performs scan line correction for Landsat 7 SLC-off images within QGIS.
It provides a user interface to select input images and output directory, then
applies chosen methods (Optimized Focal Mean or Nearest Neighbor Diffusion) to fill the gaps.

Methods Implemented:
1.  Focal Mean (Optimized using SciPy): Computes the mean of valid pixels within a defined
    window and fills the gap pixel with this mean. This implementation uses convolution
    for efficiency. Requires the 'scipy' library.
2.  Nearest Neighbor (Diffusion using GDAL): Fills gap pixels based on the values of
    nearby non-gap pixels, effectively diffusing valid data into the gaps. This
    implementation uses GDAL's optimized FillNoData function.

Instructions:
1.  Ensure you have the 'scipy' library installed in your QGIS Python environment.
    (You might need to open a OSGeo4W shell and run `pip install scipy`).
2.  Open QGIS and open the Python Console (Plugins > Python Console)
3.  Click on the "Show Editor" button in the Python Console
4.  Copy and paste this script into the editor
5.  Run the script using the "Run Script" button
"""

from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFileDialog, QLineEdit, QProgressBar,
                                QComboBox, QSpinBox, QApplication, QMessageBox)
from qgis.core import (QgsProject, QgsRasterLayer, QgsProcessingFeedback,
                      QgsProcessingContext, QgsRasterFileWriter, QgsCoordinateReferenceSystem)
import os
import processing
import numpy as np
# Use GDAL from QGIS's bundled libraries
# Assume osgeo is available in the QGIS python environment
try:
    from osgeo import gdal, osr
except ImportError:
    QMessageBox.critical(None, "Error", "GDAL/osgeo library not found. Please ensure it's installed and accessible in your QGIS Python environment.")
    raise

# Try importing SciPy for optimized methods
try:
    import scipy.ndimage
    scipy_available = True
except ImportError:
    scipy_available = False
    QMessageBox.warning(None, "Warning", "SciPy library not found. Only 'Nearest Neighbor (Diffusion)' method will be available and performant. The 'Focal Mean' method requires SciPy.")


from qgis.PyQt.QtCore import Qt
import traceback
import time # Import time for timestamp

class ScanLineCorrectionDialog(QDialog):
    def __init__(self, parent=None):
        super(ScanLineCorrectionDialog, self).__init__(parent)
        self.setWindowTitle("Landsat 7 SLC-off Scan Line Correction")
        self.resize(600, 300)
        self.setupUI()

    def setupUI(self):
        # Create layout
        layout = QVBoxLayout()

        # Input image selection
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Landsat 7 Image:")
        self.input_edit = QLineEdit()
        input_button = QPushButton("Browse...")
        input_button.clicked.connect(self.select_input)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(input_button)
        layout.addLayout(input_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        self.output_edit = QLineEdit()
        output_button = QPushButton("Browse...")
        output_button.clicked.connect(self.select_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_button)
        layout.addLayout(output_layout)

        # Method selection
        method_layout = QHBoxLayout()
        method_label = QLabel("Correction Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Focal Mean", "Nearest Neighbor (Diffusion)"]) # Simplified options
        layout.addLayout(method_layout)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)

        # Window size/Search Distance selection
        window_layout = QHBoxLayout()
        window_label = QLabel("Window Size / Max Search Distance (Pixels):") # Label updated
        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 51) # Increased max range
        self.window_spin.setValue(7) # Slightly larger default
        self.window_spin.setSingleStep(2)  # Only odd numbers (more relevant for focal)
        window_layout.addWidget(window_label)
        window_layout.addWidget(self.window_spin)
        layout.addLayout(window_layout)

        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # Buttons
        buttons_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Correction")
        self.run_button.clicked.connect(self.run_correction)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.close_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

        # Disable Focal Mean if SciPy is not available
        if not scipy_available:
            self.method_combo.model().item(0).setEnabled(False)
            self.method_combo.setCurrentIndex(1) # Select NN if FM is disabled


    def select_input(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Landsat 7 Image", "",
            "Raster Files (*.tif *.img *.tiff);;All Files (*.*)"
        )
        if filename:
            self.input_edit.setText(filename)

    def select_output(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        if directory:
            self.output_edit.setText(directory)

    def validate_inputs(self):
        # Check if input file exists
        input_file = self.input_edit.text()
        if not input_file or not os.path.isfile(input_file):
            QMessageBox.warning(self, "Error", "Please select a valid input file.")
            return False

        # Check if output directory exists
        output_dir = self.output_edit.text()
        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Error", "Please select a valid output directory.")
            return False

        # Check for SciPy if Focal Mean is selected
        if self.method_combo.currentText() == "Focal Mean" and not scipy_available:
             QMessageBox.warning(self, "Error", "SciPy library is required for the 'Focal Mean' method but was not found.")
             return False

        return True

    def run_correction(self):
        if not self.validate_inputs():
            return

        self.progress.setValue(0)
        self.run_button.setEnabled(False) # Disable button while running
        QApplication.processEvents()

        input_file = self.input_edit.text()
        output_dir = self.output_edit.text()
        correction_method = self.method_combo.currentText()
        window_size = self.window_spin.value() # Used as window size for FM, max_distance for NN

        # Get input file basename without extension
        basename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{basename}_corrected_{correction_method.lower().replace(' ', '_').replace('(', '').replace(')', '')}.tif") # More specific output name

        # Check if output file already exists and try to remove it first
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except (OSError, PermissionError) as e:
                # If can't remove, create a unique filename with timestamp
                timestamp = int(time.time())
                output_file = os.path.join(output_dir, f"{basename}_corrected_{correction_method.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.tif")
                QMessageBox.warning(
                    self, "Notice",
                    f"Could not overwrite existing file. Using alternative filename: {os.path.basename(output_file)}"
                )
                QApplication.processEvents()


        input_ds = None # Initialize outside try for cleanup
        output_ds = None # Initialize outside try for cleanup
        temp_mask_ds = None # Initialize outside try for cleanup

        try:
            # Load the input raster using GDAL
            self.progress.setValue(5)
            QApplication.processEvents()

            input_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
            if not input_ds:
                raise RuntimeError("Failed to open input file with GDAL.")

            cols = input_ds.RasterXSize
            rows = input_ds.RasterYSize
            band_count = input_ds.RasterCount
            geotransform = input_ds.GetGeoTransform()
            projection = input_ds.GetProjection()

            if band_count == 0:
                 raise RuntimeError("Input raster has no bands.")

            # Determine output data type - Float32 is safe for corrections
            output_gdal_type = gdal.GDT_Float32
            # Determine a nodata value for the output (Float32)
            # Use the input nodata if possible, convert to float
            # If input nodata is None or problematic, use a standard float nodata
            input_band1 = input_ds.GetRasterBand(1)
            input_nodata = input_band1.GetNoDataValue()
            output_nodata = float(input_nodata) if input_nodata is not None else -9999.0 # Use -9999.0 as a common float nodata

            # Create output dataset
            self.progress.setValue(10)
            QApplication.processEvents()

            driver = gdal.GetDriverByName('GTiff')
            output_ds = driver.Create(
                output_file,
                cols,
                rows,
                band_count,
                output_gdal_type
            )
            if output_ds is None:
                raise RuntimeError(f"Failed to create output dataset at {output_file}. Check permissions or if file is locked.")

            # Copy projection and geotransform
            output_ds.SetProjection(projection)
            output_ds.SetGeoTransform(geotransform)

            # Process each band
            for band_idx in range(1, band_count + 1):
                progress_start = 10 + int(80 * (band_idx - 1) / band_count)
                progress_end = 10 + int(80 * band_idx / band_count)
                self.progress.setValue(progress_start)
                QApplication.processEvents()

                # Read input band data
                input_band = input_ds.GetRasterBand(band_idx)
                data = input_band.ReadAsArray().astype(float) # Read as float array
                current_band_nodata = input_band.GetNoDataValue()

                # Determine nodata for this specific band data array
                # Use the band's nodata if available, otherwise use the default calculated earlier
                nodata_value_for_mask = float(current_band_nodata) if current_band_nodata is not None else output_nodata

                # Create mask for scan lines (1 = gap/nodata, 0 = valid data)
                # Check for exact equality with the nodata value intended to be filled
                # Use isclose for float comparison if needed, but direct == often works for typical integer nodata converted to float
                mask = np.isclose(data, nodata_value_for_mask) if np.issubdtype(data.dtype, np.floating) else (data == nodata_value_for_mask)
                mask = mask.astype(np.uint8) # Convert boolean mask to uint8 (1 or 0)


                corrected_data = np.copy(data) # Start with a copy of original data

                if correction_method == "Focal Mean":
                    if not scipy_available:
                         raise RuntimeError("SciPy is required for Focal Mean but not available.")
                    # Only apply focal mean to gap pixels
                    corrected_data[mask == 1] = self.apply_focal_mean_optimized(
                        data, mask, window_size, nodata_value_for_mask
                    )[mask == 1]

                elif correction_method == "Nearest Neighbor (Diffusion)":
                    # GDAL FillNoData works on a GDAL Band object and requires a mask band
                    # It fills pixels where mask is 0, using pixels where mask is 1
                    # Our mask is 1 for gap, 0 for valid. So we need inverse mask for GDAL: 0=fill, 1=use
                    gdal_mask_array = 1 - mask

                    # Create an in-memory GDAL dataset for the mask band
                    mask_driver = gdal.GetDriverByName('MEM')
                    # Use a name like 'mask_band' or similar, driver doesn't need a file path for MEM
                    temp_mask_ds = mask_driver.Create('', cols, rows, 1, gdal.GDT_Byte)
                    temp_mask_band = temp_mask_ds.GetRasterBand(1)
                    temp_mask_band.WriteArray(gdal_mask_array)

                    # Write the current band's original data (as float) to the output dataset first
                    output_band = output_ds.GetRasterBand(band_idx)
                    output_band.WriteArray(corrected_data) # Write the potentially Float32 data
                    output_band.SetNoDataValue(output_nodata) # Set nodata value for the output band

                    # Apply GDAL's FillNoData directly to the output band
                    # max_distance is the search radius in pixels
                    # smoothing_iterations=0 for basic nearest neighbor fill
                    # progress_callback can be used here if needed, but GDAL's isn't directly QGIS feedback
                    # Using GDAL's C API call via Python binding
                    # gdal.FillNoData(target_band, mask_band, max_distance, smoothing_iterations, options, progress_callback)
                    self.progress.setFormat(f"Processing band {band_idx}/{band_count} (NN)... %p%")
                    QApplication.processEvents()

                    # Estimate a progress step for FillNoData
                    # FillNoData doesn't give easy Python progress, so we update before/after
                    initial_progress = self.progress.value()
                    # Dummy progress update - GDAL C API call is blocking
                    # Can't easily integrate with QgsProcessingFeedback here
                    # Rely on band-by-band progress mostly
                    fill_result = gdal.FillNoData(output_band, temp_mask_band, window_size, 0, []) # No options, no callback

                    if fill_result != 0:
                        # GDAL FillNoData returns 0 on success
                         QMessageBox.warning(self, "GDAL Warning", f"GDAL FillNoData returned non-zero code {fill_result} for band {band_idx}. Check console for potential GDAL messages.")


                    # The output_band is modified in place by gdal.FillNoData
                    # We don't need to write corrected_data again for this method

                    # Clean up temporary mask dataset immediately
                    temp_mask_ds = None # This should trigger closing/deleting the in-memory dataset


                # For Focal Mean, write the corrected array (potentially modified copy)
                if correction_method == "Focal Mean":
                    output_band = output_ds.GetRasterBand(band_idx)
                    output_band.WriteArray(corrected_data)
                    output_band.SetNoDataValue(output_nodata) # Set nodata value for the output band

                output_band.FlushCache() # Ensure data is written to disk

                self.progress.setValue(progress_end)
                QApplication.processEvents()
                self.progress.setFormat(f"Processing band {band_idx}/{band_count} complete.")


            # Close datasets
            input_ds = None
            output_ds = None # This ensures the file handle is released

            # Load the corrected image into QGIS
            self.progress.setValue(95)
            QApplication.processEvents()

            layer_name = f"{basename}_corrected_{correction_method.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
            corrected_layer = QgsRasterLayer(output_file, layer_name)

            if corrected_layer.isValid():
                QgsProject.instance().addMapLayer(corrected_layer)
            else:
                QMessageBox.warning(
                    self, "Warning",
                    f"Corrected file created at {output_file} but could not be loaded into QGIS.\n"
                    "Check QGIS log messages for details."
                )

            self.progress.setValue(100)
            self.progress.setFormat("Correction complete!")
            QMessageBox.information(
                self, "Success",
                f"Scan line correction completed successfully!\nOutput saved to: {output_file}"
            )

        except Exception as e:
            # Clean up in case of error during processing
            if output_ds:
                output_ds = None # Release file handle if created

            # Attempt to remove incomplete output file
            if os.path.exists(output_file):
                try:
                    # Small delay might help if file handle is lingering
                    time.sleep(0.1)
                    os.remove(output_file)
                    print(f"Removed incomplete output file: {output_file}") # Log to console
                except Exception as cleanup_e:
                     print(f"Could not remove incomplete output file {output_file}: {cleanup_e}") # Log cleanup failure


            QMessageBox.critical(
                self, "Error",
                f"An error occurred during correction:\n{str(e)}\n\nSee Python Console for traceback."
            )
            print(f"Error details:\n{traceback.format_exc()}") # Print traceback to console

        finally:
             # Ensure button is re-enabled
             self.run_button.setEnabled(True)
             self.progress.setFormat("") # Clear format text

    def apply_focal_mean_optimized(self, data, mask, window_size, nodata_value):
        """
        Apply focal mean filter to fill scan lines gaps using optimized SciPy convolution.

        Args:
            data (np.ndarray): Input raster band data (float).
            mask (np.ndarray): Mask array (uint8), 1 for gap/nodata, 0 for valid.
            window_size (int): Size of the square focal window (e.g., 3, 5, 7).
            nodata_value (float): The nodata value used in the input data and for output gaps.

        Returns:
            np.ndarray: Array with gaps filled using focal mean.
        """
        if not scipy_available:
             # This check should theoretically be redundant due to validate_inputs,
             # but included for safety.
             print("SciPy not available, cannot run optimized focal mean.")
             return np.full_like(data, nodata_value) # Return array filled with nodata

        # Create arrays for convolution:
        # 1. Data where gaps are 0 (for summing valid values)
        # 2. Mask where valid pixels are 1 (for counting valid pixels)
        data_for_sum = np.where(mask == 0, data, 0.0)
        valid_pixel_count_mask = (1 - mask).astype(float) # 1.0 for valid, 0.0 for gap

        # Define the kernel for convolution
        kernel = np.ones((window_size, window_size))

        # Compute the sum of valid values within the window using convolution
        sum_valid = scipy.ndimage.convolve(data_for_sum, kernel, mode='constant', cval=0.0)

        # Compute the count of valid pixels within the window using convolution
        count_valid = scipy.ndimage.convolve(valid_pixel_count_mask, kernel, mode='constant', cval=0.0)

        # Calculate the mean where there are valid pixels in the window
        # Initialize with nodata, only replace where count_valid > 0
        mean_valid = np.full(data.shape, float(nodata_value), dtype=float)

        # Avoid division by zero and calculate mean only where valid neighbors exist
        has_valid_neighbors = count_valid > 0
        mean_valid[has_valid_neighbors] = sum_valid[has_valid_neighbors] / count_valid[has_valid_neighbors]

        # Create the output array, starting with the original data (as float)
        output = np.copy(data).astype(float)

        # Replace the gap pixels with the calculated mean values
        # Pixels where count_valid was 0 will be replaced with the initialized nodata value
        output[mask == 1] = mean_valid[mask == 1]

        return output

    # Removed apply_idw and apply_nearest_neighbor manual implementations


# Create and show dialog when script is run
# Check if a QApplication instance already exists (e.g., in QGIS)
app = QApplication.instance()
if not app:
    app = QApplication([]) # Create one if not
    
dlg = ScanLineCorrectionDialog()
# Run the dialog modally
dlg.exec_()

# Clean up QApplication if we created it
if 'app' in locals() and app is not QApplication.instance():
     del app