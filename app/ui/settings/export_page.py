from PySide6 import QtWidgets
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.check_box import MCheckBox
from .utils import create_title_and_combo, set_combo_box_width

class ExportPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        batch_label = MLabel(self.tr("Automatic Mode")).h4()
        batch_note = MLabel(
            self.tr(
                "Selected exports are saved to comic_translate_<timestamp> in the same directory as the input file/archive."
            )
        ).secondary()
        batch_note.setWordWrap(True)
        self.raw_text_checkbox = MCheckBox(self.tr("Export Raw Text"))
        self.translated_text_checkbox = MCheckBox(self.tr("Export Translated text"))
        self.inpainted_image_checkbox = MCheckBox(self.tr("Export Inpainted Image"))
        self.pdf_import_dpi_values = ["75", "150", "220", "300"]
        pdf_dpi_widget, self.pdf_import_dpi_combo = create_title_and_combo(
            self.tr("PDF Import Resolution"),
            [self.tr(f"{value} DPI") for value in self.pdf_import_dpi_values],
            h4=False,
        )
        set_combo_box_width(self.pdf_import_dpi_combo, [self.tr(f"{value} DPI") for value in self.pdf_import_dpi_values])
        self.pdf_import_dpi_combo.setCurrentText(self.tr("300 DPI"))
        pdf_dpi_note = MLabel(
            self.tr("Higher DPI makes imported PDF pages sharper but increases memory usage and processing time.")
        ).secondary()
        pdf_dpi_note.setWordWrap(True)

        layout.addWidget(batch_label)
        layout.addWidget(batch_note)
        layout.addWidget(self.raw_text_checkbox)
        layout.addWidget(self.translated_text_checkbox)
        layout.addWidget(self.inpainted_image_checkbox)
        layout.addWidget(pdf_dpi_widget)
        layout.addWidget(pdf_dpi_note)

        layout.addStretch(1)
