import sys
import os
import imagehash
import datetime
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings

from removedupfromfolder import move_duplicates  # Assuming this file exists and is in the same directory


class ImageDeduplicator(QThread):
    progress = Signal(int)
    finished = Signal()

    def __init__(self, input_dir, output_dir):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir

    def compute_phash(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return imagehash.phash(img)

    def run(self):
        images = [f for f in os.listdir(self.input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        hashes = {}
        duplicates = []

        find_and_filter_duplicates(self.input_dir)

        # for idx, img_name in enumerate(images):
        #     img_path = os.path.join(self.input_dir, img_name)
        #     #duplicate_count = move_duplicates(self.input_dir, dest_folder)
        #     img_hash = self.compute_phash(img_path)
        #
        #     if img_hash in hashes:
        #         duplicates.append(img_path)
        #     else:
        #         hashes[img_hash] = img_path
        #
        #     self.progress.emit(int((idx + 1) / len(images) * 100))

        os.makedirs(self.output_dir, exist_ok=True)
        for dup in duplicates:
            os.rename(dup, os.path.join(self.output_dir, os.path.basename(dup)))

        self.finished.emit()


class DeduplicationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("MyCompany", "ImageDeduplicator")  # Store settings
        self.check_trial_period()
        self.initUI()

    def check_trial_period(self):
        first_run_date_str = self.settings.value("first_run_date", None)

        if first_run_date_str is None:
            self.first_run_date = datetime.date.today()
            self.settings.setValue("first_run_date", self.first_run_date.isoformat())
        else:
            self.first_run_date = datetime.date.fromisoformat(first_run_date_str)

        self.trial_days = 20
        self.trial_ended = (datetime.date.today() - self.first_run_date).days > self.trial_days

    def initUI(self):
        self.setWindowTitle("AI moves Duplicate Images")
        self.setGeometry(200, 200, 400, 200)

        layout = QVBoxLayout()

        self.input_label = QLabel("Select Input Directory")
        layout.addWidget(self.input_label)
        self.input_button = QPushButton("Browse Input Folder")
        self.input_button.clicked.connect(self.select_input_directory)
        layout.addWidget(self.input_button)

        self.output_label = QLabel("Directory Move Duplicate Files")
        layout.addWidget(self.output_label)
        self.output_button = QPushButton("Browse Output Folder")
        self.output_button.clicked.connect(self.select_output_directory)
        layout.addWidget(self.output_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.submit_button = QPushButton("Start Deduplication")
        self.submit_button.clicked.connect(self.start_deduplication)
        layout.addWidget(self.submit_button)

        self.status_label = QLabel("Status: Waiting...")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        self.update_ui_for_trial()  # Initial UI update

    def update_ui_for_trial(self):
        if self.trial_ended:
            self.submit_button.setEnabled(False)
            self.status_label.setText(
                f"Trial period ended. Please email bigdatatech.us@gmail.com for a security key."
            )
        else:
            remaining_days = self.trial_days - (datetime.date.today() - self.first_run_date).days
            self.status_label.setText(f"Trial: {remaining_days} days remaining.")

    def select_input_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if folder:
            self.input_label.setText(f"Input: {folder}")
            self.input_dir = folder

    def select_output_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_label.setText(f"Output: {folder}")
            self.output_dir = folder

    def start_deduplication(self):
        if self.trial_ended:
            QMessageBox.warning(self, "Trial Ended", "Please email bigdatatech.us@gmail.com for a security key.")
            return

        if hasattr(self, 'input_dir') and hasattr(self, 'output_dir'):
            self.status_label.setText("Status: Processing...")
            self.progress_bar.setValue(0)

            self.worker = ImageDeduplicator(self.input_dir, self.output_dir)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self.deduplication_done)
            self.worker.start()

        else:
            self.status_label.setText("Status: Select directories first!")

    def deduplication_done(self):
        self.status_label.setText("Status: Done!")
        self.progress_bar.setValue(100)


##################################

# Identify similar images within 60 seconds and preserve the highest resolution
def find_and_filter_duplicates(folder_path):
    images = list_images(folder_path)
    duplicates = []
    image_groups = {}

    for file_path, timestamp, size in images:
        img_hash = compute_hash(file_path)
        print(img_hash)
        # Group by hash
        if img_hash in image_groups:
            image_groups[img_hash].append((file_path, timestamp, size))
        else:
            image_groups[img_hash] = [(file_path, timestamp, size)]
    print(image_groups)
    for img_hash, group in image_groups.items():
        group.sort(key=lambda x: (-x[2], x[1]))  # Sort by size (desc), then timestamp (asc)
        preserved_image = group[0]  # Keep the highest resolution image
        duplicates.extend([img[0] for img in group[1:]])  # Mark others as duplicates

    return duplicates


# Get all image files from a local folder
def list_images(folder_path):
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            timestamp = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
            size = os.path.getsize(file_path)
            images.append((file_path, timestamp, size))
    return images


# Compute hash of an image
def compute_hash(image_path):
    image = Image.open(image_path)
    return imagehash.phash(image)


###################################

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeduplicationApp()
    window.show()
    sys.exit(app.exec())