from PyQt5 import QtCore, QtGui, QtWidgets

class ImageDisplayWidget(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(ImageDisplayWidget, self).__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.scene().setBackgroundBrush(QtGui.QColor(128, 128, 128))
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        # Optionally, disable scrollbars if you want a fixed view.
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def set_image(self, image: QtGui.QImage):
        # Print image size
        print("Original Image size:", image.size())

        # Clear any existing pixmap
        self.scene().clear()

        # Create a pixmap item from the image
        pixmap = QtGui.QPixmap.fromImage(image)

        # Stretch factor (adjust as needed)
        stretch_factor = 1.0 # 1.5x wider than the original
        
        # Scale the pixmap
        scaled_pixmap = pixmap.scaled(
            int(image.width() * stretch_factor), image.height(),
            QtCore.Qt.IgnoreAspectRatio,  # This ensures stretching
            QtCore.Qt.SmoothTransformation  # Better quality scaling
        )

        # Add the transformed pixmap to the scene
        self.scene().addPixmap(scaled_pixmap)

        # Adjust scene size
        self.scene().setSceneRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())

        # Fit the view to the new image
        self.fitInView(self.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

