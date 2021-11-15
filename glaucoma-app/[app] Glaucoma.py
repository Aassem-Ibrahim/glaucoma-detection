from PyQt5 import QtCore, QtGui, QtWidgets
from qtpy import uic
from cv2 import imread
from ONH_Detection import get_cropONH, get_crop_path
from threading import Timer
from gui_utils import edit_contrast, edit_image_of, grayscale_color
from gui_utils import qImage_to_cvImage, cvImage_to_qImage, blur_qImage
from gui_utils import colorize_mask, get_boundaries_info
from shapes import Circle, Ellipse
# --------------------------------------------------------------------------- #
#                                    Colors                                   #
# --------------------------------------------------------------------------- #
COL_DISC_DEFAULT = '#ff0000'
COL_CUP_DEFAULT = '#ffaa00'
COL_PEN = '#000000'
# --------------------------------------------------------------------------- #
#                                    Other                                    #
# --------------------------------------------------------------------------- #
# Used files path
THEME_DARK = "themes/dark-theme.qss"
THEME_LIGHT = "themes/light-theme.qss"
THEME_FLAG = 'resources/theme.dat'
ABOUT_MESSAGE = 'resources/about_message.txt'
# Visual Text
T_LOADING = 'Please wait while processing'
T_SELECTED = 'Active'
T_NOTSELECTED = 'Inactive'
T_DEFINED = 'Defined'
T_UNDEFINED = 'Undefined'
T_MANUAL = 'Manual #'
T_AUTOMATIC = 'Automatic #'
T_INFO_EMPTY = '- - - - - - - -'
T_BTN_EMPTY = '- - - - -'
# Flags
MASK_DISABLED, MASK_MANUAL, MASK_AUTOMATIC = 0, 1, 2
ZOOM_DISABLED, ZOOM_OUT, ZOOM_IN = 0, 1, 2
EDIT_HUE, EDIT_SAT, EDIT_BRI = 0, 1, 2
# Shapes Parameters
OUTLINE_WIDTH = 2
CROSSHAIR_WIDTH = 2
CROSSHAIR_LENGTH = 10
VIS_THRESHOLD = 6
POINT_SIZE = 10
GRAB_AREA = 8
DEFAULT_RADII = (100, 60)
# Constants
BLUR_KSIZE = (40, 40)
CDR_THRES = 0.65


class Redraw(QtCore.QObject):
    ''' This class is responsible for emitting important QSignals '''
    # Create QSignals
    run, layer = QtCore.pyqtSignal(), QtCore.pyqtSignal()

    def image_redraw(self):
        ''' Redraw Image Viewer '''
        # Get ImageViewer reference
        img = win.qImage
        # Refresh ImageViewer
        img.refresh()
        # Enhance image using image enhancement QSliders
        win.enhance_image()
        # Repaint ImageViewer
        img.repaint()

    def activate_automatic_layer(self):
        ''' Activate disc/cup info for automatic layer '''
        # Activate both disc/cup info
        win.layers_activate_masks()
        # Update info based on automatic layer
        win.info_update_all()


def cropOHN_thread():
    ''' Thread: Crop ONH '''
    # Get ImageViewer reference
    img = win.qImage
    # Check if ImageViewer has not cropped image
    if not img.hasCrop:
        # Update hasCrop flag and get cropped region
        img.hasCrop, img.region = get_cropONH(img.filename)
        # Hide progress bar (loading)
        win.mw_end_wait_for()
        # Check if hasCrop is True after get_cropONH
        if img.hasCrop:
            # Load cropped QImage
            image = QtGui.QImage(get_crop_path(img.filename))
            # Set zoomed-in QPixmap and zoomed-in size
            img.zoomed_in = QtGui.QPixmap.fromImage(image)
            img.zoomInSize = img.zoomed_in.size().width()
            # Enable mouse tracking
            img.setMouseTracking(True)
    # Check if ImageViewer has cropped image
    if img.hasCrop:
        # Set ImageViewer zoom to zoomed-in and update QPixmap
        img.isZoomed = True
        img.pixmap = img.zoomed_in
        # Respawn shapes if not already spawned
        img.respawn()


def segmentation_thread():
    ''' Thread: Image Segmentation (mask creation) '''
    # Get ImageViewer reference
    qimg = win.qImage
    # Load MNetMask function
    from mnet_segmentation import MNetMask
    # Try to create segmented mask and get directory
    output = MNetMask(qimg.filename.split('/')[-1])
    # Check if segmentation output is a string
    if isinstance(output, str):
        # Set hasMask
        qimg.hasMask = True
        # Read mask zoomed-out
        qimg.mask_out = imread(output)
        # Activate automatic layer info (disc/cup as defined)
        qimg.redraw.layer.emit()
    else:
        # Reset hasMask
        qimg.hasMask = False
    # Hide progress bar (loading)
    win.mw_end_wait_for()
    # Redraw ImageViewer
    qimg.redraw.run.emit()


class ImageViewer(QtWidgets.QLabel):
    ''' Main interaction widget with Fundus image '''
    def __init__(self, area, replace):
        super().__init__()
        # Create signal container
        self.redraw = Redraw()
        # Connect signals with each function respectively
        self.redraw.run.connect(self.redraw.image_redraw)
        self.redraw.layer.connect(self.redraw.activate_automatic_layer)
        # Save default font
        self.default_font = replace.font()
        # Set default text and font of ImageViewer
        self.setText(replace.text())
        self.setFont(self.default_font)
        # Initialize pen (draws outline)
        self.init_pen()
        # Save parent reference (for background color changes)
        self.area = area
        # Create default QPixmap (The only image that is drawn to screen)
        self.pixmap = QtGui.QPixmap()
        # Create zoomed-in and out for quick switching
        self.zoomed_in, self.zoomed_out = QtGui.QPixmap(), QtGui.QPixmap()
        # Create hasGrab flag and grabObj reference
        self.hasGrab, self.grabObj = False, None
        # Reset all parameters
        self.reset_all()

    def reset_all(self):
        # Reset main ImageViewer variables (Flags and Images)
        self.isZoomed, self.loading = False, False
        self.hasImage, self.hasMask, self.hasCrop = False, False, False
        self.filename, self.region = None, None
        self.zoomInSize, self.zoomOutSize = None, None
        self.mask_out, self.mask_in = None, None
        self.created_mask_in = False
        # Disable mouse tracking
        self.setMouseTracking(False)

    def check_if_mask_exists(self):
        ''' Check if image has already segmented mask '''
        try:
            # Split filename
            lst = self.filename.split('/')[-2:]
            # Insert masks folder
            lst.insert(1, 'masks')
            # Create mask filename
            filename = '/'.join(lst)[:-3] + 'png'
            # Try to open the file if exists
            open(filename, 'r')
            # if open was successfuly, return True and the filename
            return True, filename
        except FileNotFoundError:
            # If file does not exist, return False and no filename
            return False, None

    def load_automatic_layer(self):
        ''' Load/Create automatic layer '''
        # Get file status and filename
        file_exists, filename = self.check_if_mask_exists()
        # Check if does file exist
        if file_exists:
            # Set hasMask and activate layer disc/cup
            self.hasMask = True
            win.layers_activate_masks()
            # Read segmented mask
            self.mask_out = imread(filename)
            # Update mask info
            win.info_update_all()
            # Redraw ImageViewer
            self.redraw.run.emit()
        else:
            # Wait for segmentation
            win.mw_wait_for('MNet Segmentation of the image')
            # Set ImageViewer loading and reset hasMask
            self.loading, self.hasMask = True, False
            # Redraw ImageViewer
            self.redraw.run.emit()
            # Run MNetSegmentation in another thread
            Timer(0.01, segmentation_thread).start()

    def toggle_zoom(self):
        ''' Set zoom level of ImageViewer '''
        # Check if ImageViewer level is zoomed-in
        if self.isZoomed:
            # Reset isZoomed
            self.isZoomed = False
            # Select ImageViewer Pixmap to zoomed_out
            self.pixmap = self.zoomed_out
            # Refresh ImageViewer background
            self.refresh()
        else:
            # Check if you have not created cropped image
            if not self.hasCrop:
                # Wait for ONH_Detection and set loading
                win.mw_wait_for('Detecting Fundus location in the image')
                self.loading = True
                # Start cropOHN thread
                Timer(0.001, cropOHN_thread).start()
            else:
                # Set isZoomed and select ImageViewer Pixmap to zoomed_in
                self.isZoomed = True
                self.pixmap = self.zoomed_in
                # Refresh ImageViewer background
                self.refresh()
                # Redraw ImageViewer
                self.repaint()
        # Set focus to force repainting
        self.setFocus(True)

    def get_image_file(self):
        # Open FileDialog to get an image
        qfd = QtWidgets.QFileDialog
        filename, _ = qfd.getOpenFileName(self,
                                          "Select Glaucoma Case",
                                          "glaucoma-cases/",
                                          "Image Files (*.jpg)")
        # Check if file has been selected
        if filename != '':
            # Make sure you have not selected a cropped image
            if filename[-8:] == '_mod.jpg':
                # Show error message for selecting a cropped image
                win.show_error('Image Load Error',
                               '<p>You cannot load <b>cropped image</b>.</p>')
            else:
                # Set hasChanged
                win.isChanged = True
                # Set hasImage and save filename and create a QImage
                self.hasImage, self.filename = True, filename
                qimage = QtGui.QImage(filename)
                # Enable Image Enhancement, Zoom and Layers blocks
                win.imageEnhancement_setEnabled(True)
                win.zoom_setMode(ZOOM_OUT)
                win.layers_setEnabled(True)
                win.create_add_menu()
                # Set zoomed_out and ImageViewer QPixmap
                self.zoomed_out = QtGui.QPixmap.fromImage(qimage)
                self.pixmap = self.zoomed_out
                # Get zoomOutSize
                self.zoomOutSize = self.zoomed_out.size().width()
                # Refresh ImageViewer background
                self.refresh()
                # Enhance ImageViewer default QPixmap
                win.enhance_image()

    def forward_transformation(self, shape):
        ''' Project shape onto screen '''
        # Check if isZoommed
        if self.isZoomed:
            # Calculate scale factor
            factor = self.imageSize/self.zoomInSize
            # Transform the shape (scale and move)
            shape.scale(factor)
            shape.move(self.xdiff, self.ydiff)
        else:
            # Calculate scale factor
            factor = self.imageSize/self.zoomOutSize
            # Calculate region difference for zoomed-out image
            rxdiff = round(self.region[0] * factor)
            rydiff = round(self.region[1] * factor)
            # Transform the shape (scale and move)
            shape.scale(factor)
            shape.move(self.xdiff + rxdiff,
                       self.ydiff + rydiff)
        # Return Transformed shape
        return shape

    def inverse_tranformation(self, mx, my):
        ''' Project point from screen to original size '''
        # Check if isZoomed
        if self.isZoomed:
            # Calculate factor
            factor = self.zoomInSize / self.imageSize
            # Calculate tx, ty (final transformed point)
            tx, ty = mx - self.xdiff, my - self.ydiff
        else:
            # Calculate factor
            factor = self.zoomOutSize / self.imageSize
            # Calculate region differences
            rxdiff, rydiff = self.region[0] / factor, self.region[1] / factor
            # Calculate tx, ty (final transformed point)
            tx, ty = mx - (self.xdiff + rxdiff), my - (self.ydiff + rydiff)
        # Return scaled tranformed point (last transformation step)
        return round(tx * factor), round(ty * factor)

    def get_layer_points(self):
        ''' Get points of current selected layer projected onto screen '''
        # Create empty lists for centers and radii
        centers, radii = [], []
        # Make sure there is a layer that is selected
        if win.current != -1:
            # Loop over disc/cup of current layer
            for dsc_cup, shape in enumerate(win.shapes[win.current]):
                # Check if shape is a circle
                if isinstance(shape, Circle):
                    # Get transformed shape using a copy
                    transCircle = self.forward_transformation(shape.copy())
                    # Add circle center point to centers
                    centers.append((dsc_cup, transCircle.c))
                    # Add circle radius point to radii
                    radii.append((dsc_cup, transCircle.r))
                # Check if shape is an ellipse
                elif isinstance(shape, Ellipse):
                    # TODO: Not implemented yet
                    pass
        # Return centers and radii points
        return centers, radii

    def get_visibility_flag(self, dsc_cup):
        ''' Get visibility flag of disc/cup layer '''
        # Check if dsc_cup variable is disc
        if dsc_cup == 0:
            # Return True if disc_alpha slider value is bigger than threshold
            return win.qS_disc_alpha.value() > VIS_THRESHOLD
        else:
            # Return True if cup_alpha slider value is bigger than threshold
            return win.qS_cup_alpha.value() > VIS_THRESHOLD

    def setGrabObject(self, mx, my):
        ''' Grab object if mouse(x, y) is in range '''
        # Get layer points
        centers, radii = self.get_layer_points()
        # Loop over center points
        for dsc_cup, center in centers:
            # Check if center is in mouse(x, y) range
            if center.isIn(mx, my, GRAB_AREA):
                # Check if points are visible to be selected
                if self.get_visibility_flag(dsc_cup):
                    # Set hasGrab
                    self.hasGrab = True
                    # Set grabObj reference
                    self.grabObj = win.shapes[win.current][dsc_cup].c
                    # Exit function
                    return
        # Loop over radius points
        for dsc_cup, radius in radii:
            # Check if center is in mouse(x, y) range
            if radius.isIn(mx, my, GRAB_AREA):
                # Check if points are visible to be selected
                if self.get_visibility_flag(dsc_cup):
                    # Set hasGrab
                    self.hasGrab = True
                    # Set grabObj reference
                    self.grabObj = win.shapes[win.current][dsc_cup].r
                    # Exit function
                    return

    def setHoverCursor(self, mx, my):
        ''' Change mouse to pointing hand if mouse(x, y) is in range '''
        # Get layer points
        centers, radii = self.get_layer_points()
        # Loop over center points
        for dsc_cup, center in centers:
            # Check if center is in mouse(x, y) range
            if center.isIn(mx, my, GRAB_AREA):
                # Check if points are visible to be selected
                if self.get_visibility_flag(dsc_cup):
                    # Set mouse cursor to pointing hand
                    self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                    # Exit function
                    return
        # Loop over radius points
        for dsc_cup, radius in radii:
            # Check if center is in mouse(x, y) range
            if radius.isIn(mx, my, GRAB_AREA):
                # Check if points are visible to be selected
                if self.get_visibility_flag(dsc_cup):
                    # Set mouse cursor to pointing hand
                    self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                    # Exit function
                    return
        # Set mouse cursor to arrow
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    def mousePressEvent(self, event):
        ''' ImageViewer MousePressEvent '''
        # Check if image is not loaded yet
        if not self.hasImage:
            # Get image
            self.get_image_file()
        else:
            # Set grabObj if mouse(x, y) over a point
            self.setGrabObject(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        ''' ImageViewer mouseReleaseEvent '''
        # Reset hasGrab
        self.hasGrab = False

    def mouseMoveEvent(self, event):
        ''' ImageViewer mouseMoveEvent '''
        # Get mouse(x, y)
        mx, my = event.x(), event.y()
        # Check if not hasGrab
        if not self.hasGrab:
            # Set hover cursor if mouse(x, y) over a point
            self.setHoverCursor(mx, my)
        else:
            # Set grabObj position by inverse transformation of mouse(x, y)
            self.grabObj.x, self.grabObj.y = self.inverse_tranformation(mx, my)
            # Redraw ImageViewer
            self.repaint()
            # Update info of current layer
            win.info_update_all()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        if self.loading or self.hasImage:
            labelSize = self.size()
            scaledPix = self.pixmap.scaled(labelSize,
                                           QtCore.Qt.KeepAspectRatio,
                                           QtCore.Qt.SmoothTransformation)
            imageSize = scaledPix.size()

            x1, y1 = labelSize.width(), labelSize.height()
            x2, y2 = imageSize.width(), imageSize.height()

            self.xdiff, self.ydiff = (x1 - x2) // 2, (y1 - y2) // 2
            self.imageSize = x2

            point = QtCore.QPoint(self.xdiff, self.ydiff)

            if self.loading:
                self.setLoading(painter, point, scaledPix)
            elif self.hasImage:
                painter.drawPixmap(point, scaledPix)
                if win.current != -1:
                    layer = win.get_layer_type(win.current)
                    if layer:
                        self.draw_shapes(painter)
                    else:
                        self.show_mask(painter, point)
        else:
            painter.setBrush(QtGui.QColor(COL_FRAME))
            painter.drawText(event.rect(),
                             QtCore.Qt.AlignCenter,
                             self.text())

    def show_mask(self, painter, point):
        if self.hasMask:
            labelSize = self.size()
            disc_alpha = win.qS_disc_alpha.value()
            cup_alpha = win.qS_cup_alpha.value()

            if self.isZoomed:
                if self.hasCrop is True and self.created_mask_in is False:
                    self.created_mask_in = True
                    x, y, xf, yf = self.region
                    w, h = xf - x, yf - y
                    self.mask_in = self.mask_out[y:y+h, x:x+w].copy()

                qi = colorize_mask(self.mask_in, None,
                                   disc_alpha, cup_alpha,
                                   win.disc_color, win.cup_color)
            else:
                scale = min(labelSize.width(), labelSize.height())
                qi = colorize_mask(self.mask_out, scale,
                                   disc_alpha, cup_alpha,
                                   win.disc_color, win.cup_color)

            pixmap = QtGui.QPixmap(qi)
            scaledPix = pixmap.scaled(labelSize,
                                      QtCore.Qt.KeepAspectRatio,
                                      QtCore.Qt.SmoothTransformation)
            painter.drawPixmap(point, scaledPix)

    def setLoading(self, painter, point, pixmap):
        msg = T_LOADING

        pixmap = QtGui.QPixmap(blur_qImage(pixmap, BLUR_KSIZE))
        painter.drawPixmap(point, pixmap)

        rect = painter.fontMetrics().boundingRect(msg)
        ppp = QtCore.QPoint((self.width() - rect.width())//2,
                            (self.height() + rect.height())//2)

        painterPath = QtGui.QPainterPath()
        painterPath.addText(ppp, self.default_font, msg)
        painter.strokePath(painterPath, QtGui.QPen(QtGui.QColor('#000000'), 2))
        painter.fillPath(painterPath, QtGui.QColor('#FFFFFF'))
        painter.end()

    def draw_shapes(self, painter):
        para = [[win.qCB_disc_outline.isChecked(),
                 win.qS_disc_alpha.value(),
                 QtGui.QColor(win.disc_color)],
                [win.qCB_cup_outline.isChecked(),
                 win.qS_cup_alpha.value(),
                 QtGui.QColor(win.cup_color)]]

        para[0][2].setAlpha(para[0][1])
        para[1][2].setAlpha(para[1][1])

        centers, radii = [], []

        if win.current != -1:
            for i, shape in enumerate(win.shapes[win.current]):
                if isinstance(shape, Circle):
                    transCircle = shape.copy()
                    if self.isZoomed:
                        transCircle.scale(self.imageSize / self.zoomInSize)
                        transCircle.move(self.xdiff, self.ydiff)
                    elif self.region is not None:
                        factor = self.imageSize / self.zoomOutSize
                        xxdiff = int(self.region[0] * factor)
                        yydiff = int(self.region[1] * factor)
                        transCircle.scale(factor)
                        transCircle.move(self.xdiff + xxdiff,
                                         self.ydiff + yydiff)
                    centers.append((transCircle.c.value(), para[i][1]))
                    radii.append((transCircle.r.value(), para[i][1]))
                    self.draw_circle(painter,
                                     transCircle,
                                     para[i][0],
                                     para[i][1],
                                     para[i][2])

            for center in centers:
                self.draw_crosshair(painter, center[0], center[1])

            for radius in radii:
                self.draw_point(painter, radius[0], radius[1])

    def draw_circle(self, painter, circle, cb, sld, col):
        dia = circle.dia()
        rad = dia // 2
        (cx, cy) = circle.center()
        (rx, ry) = circle.radius()
        pen = self.qOutlinePen if sld > VIS_THRESHOLD else self.qNoPen

        if cb:
            self.qOutlinePen.setWidth(OUTLINE_WIDTH)
            painter.setPen(pen)
        else:
            painter.setPen(self.qNoPen)

        painter.setBrush(col)
        painter.drawEllipse(cx-rad, cy-rad, dia, dia)

    def draw_crosshair(self, painter, point, sld):
        if sld > VIS_THRESHOLD:
            cx, cy = point
            cross = CROSSHAIR_LENGTH // 2
            self.qOutlinePen.setWidth(CROSSHAIR_WIDTH)
            painter.setPen(self.qOutlinePen)
            painter.drawLine(cx-cross, cy, cx+cross, cy)
            painter.drawLine(cx, cy-cross, cx, cy+cross)

    def draw_point(self, painter, point, sld):
        if sld > VIS_THRESHOLD:
            cx, cy = point
            rad = POINT_SIZE // 2
            self.qOutlinePen.setWidth(1)
            painter.setPen(self.qOutlinePen)
            painter.setBrush(self.pen_col)
            painter.drawEllipse(cx-rad, cy-rad, POINT_SIZE, POINT_SIZE)

    def init_pen(self):
        self.no_pen = QtGui.QColor()
        self.no_pen.setAlpha(0)
        self.qNoPen = QtGui.QPen(self.no_pen)

        self.pen_col = QtGui.QColor(COL_PEN)
        self.pen_col.setAlpha(200)
        self.qOutlinePen = QtGui.QPen(self.pen_col)

    def respawn(self):
        if win.current != -1 and self.hasCrop:
            for i, shape in enumerate(win.shapes[win.current]):
                if isinstance(shape, Circle):
                    if not shape.used:
                        shape.used = True
                        (cx, cy) = 256, 256
                        (rx, ry) = (256 + DEFAULT_RADII[i], 256)
                        (shape.c.x, shape.c.y) = cx, cy
                        (shape.r.x, shape.r.y) = rx, ry
        win.info_update_all()
        self.redraw.run.emit()

    def refresh(self):
        if self.hasImage:
            if not self.isZoomed and win.theme:
                self.area.setStyleSheet('background-color: black;')
            elif self.isZoomed:
                self.area.setStyleSheet(f'background-color: {COL_FRAME};')
        else:
            self.area.setStyleSheet(f'background-color: {COL_FRAME};')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ui_file, parent=None):
        super(MainWindow, self).__init__(parent)

        self.theme = self.load_theme_flag()
        self.set_dark_mode(self.theme)

        uic.loadUi(ui_file, self)

        self.menu_set_dark_mode(self.theme)
        self.connect_signals()
        self.create_main_variables()

        self.qImage = ImageViewer(area=self.qV, replace=self.qImage)
        self.qV.setWidget(self.qImage)

        self.qTW_layers.setMaximumSize(QtCore.QSize(280, 16777215))
        self.qTW_layers.header().resizeSection(0, 174)

        for qPB in (self.qPB_reset_hue, self.qPB_reset_brightness,
                    self.qPB_reset_contrast, self.qPB_reset_saturation):
            qPB.setStyleSheet('QPushButton{border:none; background:#000000FF;}'
                              'QPushButton:pressed{background: #ACB1B6;}')

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setRange(0, 1)
        self.progressBar.hide()
        self.progressBarLabel = QtWidgets.QLabel("")
        self.progressBarLabel.setStyleSheet("QLabel { font-size: 16px; }")
        self.progressBarLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBarLabel.hide()

        self.statusBar().addPermanentWidget(self.progressBarLabel, 1)
        self.statusBar().addPermanentWidget(self.progressBar, 1)

        self.showMaximized()

        Timer(0.1, load_models).start()

    def mw_wait_for(self, msg):
        self.progressBarLabel.setText(msg)
        self.progressBarLabel.show()
        self.progressBar.setRange(0, 0)
        self.progressBar.show()

    def mw_end_wait_for(self):
        self.progressBarLabel.setText('')
        self.progressBarLabel.hide()
        self.progressBar.setRange(0, 1)
        self.progressBar.hide()
        self.qImage.loading = False

    def create_main_variables(self):
        # set default mask colors
        self.disc_color = COL_DISC_DEFAULT
        self.cup_color = COL_CUP_DEFAULT
        # initialize corners
        self.layers_setEnabled(False)
        self.mask_setMode(MASK_DISABLED)
        self.mask_reset_buttons()
        self.imageEnhancement_setEnabled(False)
        self.info_setEnabled(False)
        self.info_reset()
        self.zoom_setMode(ZOOM_DISABLED)
        self.mw_refresh_frames()
        # QButtonGroup
        self.qBG_layers = QtWidgets.QButtonGroup()
        self.qBG_layers.buttonToggled.connect(self.layer_changed)
        # variables
        self.layers = [0, 0]
        self.shapes = []
        self.mask_layers = [False, False]
        self.current = -1
        self.isChanged = False

    def connect_signals(self):
        # ------------------------------------------------------------------- #
        #                               Signals                               #
        # ------------------------------------------------------------------- #
        # Buttons Signals
        self.qPB_remove.clicked.connect(self.btn_remove_click)
        self.qPB_disc.clicked.connect(self.btn_disc)
        self.qPB_cup.clicked.connect(self.btn_cup)
        self.qPB_zoom_in.clicked.connect(self.btn_zoom_in)
        self.qPB_zoom_out.clicked.connect(self.btn_zoom_out)
        self.connect_image_enhancement_reset_buttons()
        # QFrame Events
        self.qPB_disc_color.mousePressEvent = self.btn_disc_color
        self.qPB_cup_color.mousePressEvent = self.btn_cup_color
        # QCheckBox Signals
        self.qCB_cup_outline.stateChanged.connect(self.cb_cup_outline)
        self.qCB_disc_outline.stateChanged.connect(self.cb_disc_outline)
        # QSlider Signals
        self.qS_disc_alpha.valueChanged.connect(self.s_disc_alpha)
        self.qS_cup_alpha.valueChanged.connect(self.s_cup_alpha)
        self.qS_hue.valueChanged.connect(self.s_hue)
        self.qS_brightness.valueChanged.connect(self.s_brightness)
        self.qS_saturation.valueChanged.connect(self.s_saturation)
        self.qS_contrast.valueChanged.connect(self.s_contrast)
        # QMenu Signals
        self.actionNew.triggered.connect(self.menu_new)
        self.actionLoad.triggered.connect(self.menu_open)
        self.actionSave.triggered.connect(self.menu_save)
        self.actionExit.triggered.connect(self.menu_exit)
        self.actionDarkMode.triggered.connect(self.menu_dark_mode)
        self.actionAbout.triggered.connect(self.menu_about)
        self.actionAboutQt.triggered.connect(self.menu_about_qt)
        self.actionGitHub.triggered.connect(self.menu_github)

    def connect_image_enhancement_reset_buttons(self):
        # Get Reset QPushButton references
        pbr_HUE, pbr_BRT = self.qPB_reset_hue, self.qPB_reset_brightness
        pbr_SAT, pbr_CTRS = self.qPB_reset_saturation, self.qPB_reset_contrast
        # Get QSlider references
        s_HUE, s_BRT = self.qS_hue, self.qS_brightness
        s_SAT, s_CTRS = self.qS_saturation, self.qS_contrast
        # Connect signals to reset
        pbr_HUE.clicked.connect(lambda: s_HUE.setValue(-255))
        pbr_BRT.clicked.connect(lambda: s_BRT.setValue(0))
        pbr_SAT.clicked.connect(lambda: s_SAT.setValue(0))
        pbr_CTRS.clicked.connect(lambda: s_CTRS.setValue(0))

    def create_add_menu(self):
        menu = QtWidgets.QMenu(self.qPB_add,
                               triggered=self.on_menu_triggered)
        act1 = menu.addAction("Automatic Mask")
        act1.setIcon(QtGui.QIcon("resources/menu_automatic.png"))
        act2 = menu.addAction("Manual Mask")
        act2.setIcon(QtGui.QIcon("resources/menu_manual.png"))
        self.qPB_add.setMenu(menu)

    def remove_add_menu(self):
        self.qPB_add.setMenu(None)

    @QtCore.pyqtSlot(QtWidgets.QAction)
    def on_menu_triggered(self, action):
        if action.text() == "Manual Mask":
            self.add_manual_mask()
            self.isChanged = True
        elif action.text() == "Automatic Mask":
            self.add_automatic_mask()
            self.isChanged = True

    def load_theme_flag(self):
        try:
            with open(THEME_FLAG) as f:
                data = int(f.read())
            return data == 1
        except FileNotFoundError:
            return True

    def save_theme_flag(self):
        with open(THEME_FLAG, 'w') as f:
            f.write('1' if self.theme else '0')

    def add_manual_mask(self):
        ok, text = self.ask_for_layer_name()
        if ok:
            self.layers_add(True, text)

    def add_automatic_mask(self):
        ok, text = self.ask_for_layer_name()
        if ok:
            self.layers_add(False, text)

    def ask_for_layer_name(self):
        qin = QtWidgets.QInputDialog
        text, ok = qin.getText(self,
                               'Layer Name',
                               'Enter layer name:')
        return (ok, text)

    def set_dark_mode(self, flag, refresh=False):
        self.theme = flag
        self.load_colors(flag)
        theme_filename = THEME_DARK if flag else THEME_LIGHT
        file = QtCore.QFile(theme_filename)
        file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text)
        stream = QtCore.QTextStream(file)
        if refresh:
            self.reset_frames_style()
        app.setStyleSheet(stream.readAll())
        if refresh:
            self.refresh_all()

    def refresh_all(self):
        self.qImage.refresh()
        self.mw_refresh_frames()
        self.update_radio_buttons()
        self.update_layers_info()
        self.info_update_all()

    def load_colors(self, flag):
        global COL_FG_DEF, COL_BG_DEF, COL_FG_UNDEF,\
               COL_BG_UNDEF, COL_FG_P_SEL, COL_BG_P_SEL,\
               COL_FG_C_SEL, COL_BG_C_SEL, COL_FAIL, COL_PASS,\
               COL_FRAME
        if flag:
            COL_FG_DEF = '#ffffff'
            COL_BG_DEF = '#005400'
            COL_FG_UNDEF = '#ffffff'
            COL_BG_UNDEF = '#540000'
            COL_FG_P_SEL = '#ffffff'
            COL_BG_P_SEL = '#666600'
            COL_FG_C_SEL = '#ffffff'
            COL_BG_C_SEL = '#545400'
            COL_FAIL = '#dd2222'
            COL_PASS = '#22aa22'
            COL_FRAME = '#29333D'
        else:
            COL_FG_DEF = '#000000'
            COL_BG_DEF = '#97ff9b'
            COL_FG_UNDEF = '#000000'
            COL_BG_UNDEF = '#ff9e9e'
            COL_FG_P_SEL = '#000000'
            COL_BG_P_SEL = '#ecec8c'
            COL_FG_C_SEL = '#000000'
            COL_BG_C_SEL = '#ffffac'
            COL_FAIL = '#AA2233'
            COL_PASS = '#22AA33'
            COL_FRAME = '#EAEAEA'

    def showEvent(self, event):
        self.mw_wait_for('Loading Deep NN models')

    def info_update_all(self):
        if all(self.mask_layers):
            layer = self.get_layer_type(win.current)
            if layer:
                shapes = self.shapes[self.current]
                disc, cup = shapes[0], shapes[1]

                qimg = self.qImage
                rx, ry = qimg.region[0], qimg.region[1]

                (dx, dy), dd = disc.c.value(), disc.dia()
                (cx, cy), cd = cup.c.value(), cup.dia()
                dr, cr = dd / 2, cd / 2
                dx, dy, cx, cy = dx-rx, ry-dy, cx-rx, ry-cy
                norm = dd

                n = round(dx + dr - cx - cr)
                i = round(dy + dr - cy - cr)
                s = round(cy - cr - dy + dr)
                t = round(cx - cr - dx + dr)

                cdr = cd/dd

                cx, cy = cx - dx + dr, cy - dy + dr
                cx, cy = cx / norm, 1 - (cy / norm)
                dx, dy = 0.5, 0.5
                dd, cd = dd / norm, cd / norm
                cw, ch, dw, dh = cd, cd, dd, dd
                da, ca = 0.785398, 0.785398 * cdr * cdr

            else:
                img = self.qImage.mask_out
                gbi = get_boundaries_info
                (((cx, cy, cw, ch), ca), ((dx, dy, dw, dh), da)) = gbi(img)
                disc = {'x': dx, 'y': dy, 'w': dw, 'h': dh, 'a': da}
                cup = {'x': cx, 'y': cy, 'w': cw, 'h': ch, 'a': ca}

                data = normalize_data({'disc': disc, 'cup': cup})

                dw2, dh2 = dw / 2, dh / 2
                cw2, ch2 = cw / 2, ch / 2
                cx, cy = dx + dw2 - dx, dy + dh2 - dy
                dx, dy = dw / 2, dh / 2
                dd, cd = max(dw, dh), max(cw, ch)
                norm = max(dd, cd)
                cdr = ch/dh

                n = (dx + dw2 - cx - cw2)
                i = (dy + dh2 - cy - ch2)
                s = (cy - ch2 - dy + dh2)
                t = (cx - cw2 - dx + dw2)

                dx, dy = data['disc']['x'], data['disc']['y']
                dw, dh = data['disc']['w'], data['disc']['h']
                da, ca = data['disc']['a'], data['cup']['a']
                cx, cy = data['cup']['x'], data['cup']['y']
                cw, ch = data['cup']['w'], data['cup']['h']

            if dd < cd:
                self.set_isnt(T_INFO_EMPTY)
                self.set_cdr('Cup > Disc')
                self.set_dtr(T_INFO_EMPTY)
            elif n < 0 or i < 0 or s < 0 or t < 0:
                self.set_isnt('Cup is out')
                self.set_cdr(T_INFO_EMPTY)
                self.set_dtr(T_INFO_EMPTY)
            else:
                self.set_isnt((i >= s, s >= n, n >= t))
                self.set_cdr(cdr)

                if cdr_model is not None:
                    prediction = cdr_model.predict([cx, cy, cw, ch, ca,
                                                    dx, dy, dw, dh, da])
                    if prediction[0][0] != -1:
                        self.set_dtr(float(prediction[0][0]))
                    else:
                        self.set_dtr('Model Error!')
                else:
                    self.set_dtr('Load Error!')
        else:
            self.info_reset()

    def enhance_image(self):
        qimg = self.qImage

        sHue = (self.qS_hue.value() + 255) // 2
        sSat = self.qS_saturation.value()
        sBri = self.qS_brightness.value()
        sCon = self.qS_contrast.value()

        qpix = qimg.zoomed_in if qimg.isZoomed else qimg.zoomed_out

        cv2i = qImage_to_cvImage(qpix)
        cv2i = edit_contrast(cv2i, sCon)
        cv2i = edit_image_of(cv2i, EDIT_HUE, sHue)
        cv2i = edit_image_of(cv2i, EDIT_SAT, sSat)
        cv2i = edit_image_of(cv2i, EDIT_BRI, sBri)
        qimg.pixmap = QtGui.QPixmap(cvImage_to_qImage(cv2i))

    def show_error(self, title, info, more=''):
        qm = QtWidgets.QMessageBox
        msg = qm(self)
        msg.setText(info)
        msg.setIcon(qm.Critical)
        msg.setWindowTitle(title)
        msg.setInformativeText(more)
        msg.exec_()

    def fix_layers_name(self):
        automatic, manual = 0, 0
        for i in range(sum(self.layers)):
            item = self.qTW_layers.topLevelItem(i)
            text = item.text(0).split('#')
            layer_name = text[1].split(':')
            layer_type_str, idx = text[0][0], int(layer_name[0])
            layer_type = layer_type_str == T_MANUAL[0]
            if layer_type:
                manual += 1
                out_of_order = (idx != manual)
            else:
                automatic += 1
                out_of_order = (idx != automatic)

            if out_of_order:
                name = T_MANUAL if layer_type else T_AUTOMATIC
                idx = manual if layer_type else automatic
                title = ''.join(layer_name[1:])
                title = f':{title}' if title != '' else ''
                item.setText(0, f'{name}{idx}{title}')

    def get_selected_layer_index(self):
        for i in range(sum(self.layers)):
            item = self.qTW_layers.topLevelItem(i)
            btn = self.qTW_layers.itemWidget(item, 1)
            if btn.isChecked():
                self.current = i
                return
        self.current = -1

    def btn_remove_click(self):
        idx = self.current

        if idx != -1:
            item = self.qTW_layers.topLevelItem(idx)
            text = item.text(0)
            qm = QtWidgets.QMessageBox
            ans = qm.question(self,
                              'Layer Removal Confirmation',
                              'Are you sure to delete this layer?'
                              f'\n{text}'
                              '\n\nNOTE: This process cannot be undone',
                              qm.Yes | qm.No)

            if ans == qm.Yes:
                button = self.qTW_layers.itemWidget(item, 1)
                self.qBG_layers.removeButton(button)
                layer_letter = self.qTW_layers.takeTopLevelItem(idx).text(0)[0]
                if layer_letter == T_MANUAL[0]:
                    self.layers[0] -= 1
                else:
                    self.layers[1] -= 1

                layers_count = sum(self.layers)
                if layers_count:
                    count = layers_count
                    msg = f'A layer has been selected from ({count}) layers!'
                    idx = idx - 1 if layers_count == idx else idx
                    del self.shapes[self.current]
                    self.current = idx
                    item = self.qTW_layers.topLevelItem(idx)
                    self.qTW_layers.itemWidget(item, 1).click()
                else:
                    msg = 'No layers are left!'
                    self.qPB_remove.setEnabled(False)
                    self.info_setEnabled(False)
                    self.info_reset()
                    self.mask_setMode(MASK_DISABLED)
                    self.mask_reset_buttons()
                    self.current = -1

                self.fix_layers_name()
                qm.information(self,
                               'Layer Removal Status',
                               'The layer has been removed successfully'
                               f'\n\n{msg}',
                               qm.Ok)

    def COL_dialog(self, color, button):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(color), self)
        if col.isValid():
            col = col.name(0)
            button.setStyleSheet(f"background-color: {col};")
            return col
        return color

    def btn_disc_color(self, event):
        if self.mask_disc:
            self.disc_color = self.COL_dialog(self.disc_color,
                                              self.qPB_disc_color)
            self.qImage.repaint()

    def btn_cup_color(self, event):
        if self.mask_cup:
            self.cup_color = self.COL_dialog(self.cup_color,
                                             self.qPB_cup_color)
            self.qImage.repaint()

    def ask_removal_of(self, mask):
        qm = QtWidgets.QMessageBox
        ans = qm.question(self,
                          'Mask Removal Confirmation',
                          f'<p>Are you sure to delete <b>{mask} mask</b>?</p>'
                          '<p><b>NOTE:</b> This process cannot be undone</p>',
                          qm.Yes | qm.No)

        if ans == qm.Yes:
            return True
        else:
            return False

    def btn_disc(self):
        if self.mask_layers[0] and self.ask_removal_of('disc'):
            self.set_layer_child(0, False)
            self.shapes[self.current][0] = None
            self.info_update_all()

    def btn_cup(self):
        if self.mask_layers[1] and self.ask_removal_of('cup'):
            self.set_layer_child(1, False)
            self.shapes[self.current][1] = None
            self.info_update_all()

    def btn_zoom_in(self):
        self.zoom_setMode(ZOOM_IN)
        self.qImage.toggle_zoom()
        if self.qImage.zoomed_out is not None:
            self.enhance_image()

    def btn_zoom_out(self):
        self.zoom_setMode(ZOOM_OUT)
        self.qImage.toggle_zoom()
        if self.qImage.zoomed_in is not None:
            self.enhance_image()

    def cb_disc_outline(self):
        self.qImage.repaint()

    def cb_cup_outline(self):
        self.qImage.repaint()

    def s_disc_alpha(self):
        self.qImage.repaint()

    def s_cup_alpha(self):
        self.qImage.repaint()

    def s_hue(self):
        self.enhance_image()
        self.qImage.repaint()

    def s_brightness(self):
        self.enhance_image()
        self.qImage.repaint()

    def s_saturation(self):
        self.enhance_image()
        self.qImage.repaint()

    def s_contrast(self):
        self.enhance_image()
        self.qImage.repaint()

    def mask_rename_buttons(self, disc=None, cup=None):
        if isinstance(disc, str):
            self.qPB_disc.setText(disc)
        if isinstance(cup, str):
            self.qPB_cup.setText(cup)

    def mask_reset_buttons(self):
        self.mask_rename_buttons(disc=T_BTN_EMPTY,
                                 cup=T_BTN_EMPTY)
        self.qPB_disc.setMenu(None)
        self.qPB_cup.setMenu(None)

    def mask_disc_setEnabled(self, flag):
        self.mask_disc = flag
        self.qS_disc_alpha.setEnabled(flag)
        self.qCB_disc_outline.setEnabled(flag)
        if not flag:
            d_col = grayscale_color(self.disc_color)
            cursor = QtGui.QCursor(QtCore.Qt.ArrowCursor)
        else:
            d_col = self.disc_color
            cursor = QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        self.qPB_disc_color.setStyleSheet(f"background-color: {d_col};")
        self.qPB_disc_color.setCursor(cursor)

    def mask_cup_setEnabled(self, flag):
        self.mask_cup = flag
        self.qS_cup_alpha.setEnabled(flag)
        self.qCB_cup_outline.setEnabled(flag)
        if not flag:
            c_col = grayscale_color(self.cup_color)
            cursor = QtGui.QCursor(QtCore.Qt.ArrowCursor)
        else:
            c_col = self.cup_color
            cursor = QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        self.qPB_cup_color.setStyleSheet(f"background-color: {c_col};")
        self.qPB_cup_color.setCursor(cursor)

    def mask_setMode(self, mode):
        self.mask_mode = mode
        if mode == MASK_DISABLED:
            self.qL_disc.setEnabled(False)
            self.qL_cup.setEnabled(False)
            self.qPB_disc.setEnabled(False)
            self.qPB_cup.setEnabled(False)
            self.mask_disc_setEnabled(False)
            self.mask_cup_setEnabled(False)
            self.mask_disc = False
            self.mask_cup = False
        else:
            self.qL_disc.setEnabled(True)
            self.qL_cup.setEnabled(True)
            self.mask_disc_setEnabled(True)
            self.mask_cup_setEnabled(True)
            self.mask_disc = True
            self.mask_cup = True
            if mode == MASK_AUTOMATIC:
                self.qPB_disc.setEnabled(False)
                self.qPB_cup.setEnabled(False)
            else:
                self.qPB_disc.setEnabled(True)
                self.qPB_cup.setEnabled(True)

    def info_setEnabled(self, flag):
        self.info_enabled = flag
        self.qL_isnt.setEnabled(flag)
        self.qL_isnt_val.setEnabled(flag)
        self.qL_cdr.setEnabled(flag)
        self.qL_cdr_val.setEnabled(flag)
        self.qL_dtr.setEnabled(flag)
        self.qL_dtr_val.setEnabled(flag)

    def info_reset(self):
        self.qL_isnt_val.setText(T_INFO_EMPTY)
        self.qL_cdr_val.setText(T_INFO_EMPTY)
        self.qL_dtr_val.setText(T_INFO_EMPTY)
        self.qL_isnt_val.setStyleSheet('')
        self.qL_cdr_val.setStyleSheet('')
        self.qL_dtr_val.setStyleSheet('')

    def zoom_setMode(self, mode):
        self.zoom_mode = mode
        if mode == ZOOM_DISABLED:
            self.qL_zoom.setEnabled(False)
            self.qPB_zoom_in.setEnabled(False)
            self.qPB_zoom_out.setEnabled(False)
        elif mode == ZOOM_OUT:
            self.qL_zoom.setEnabled(True)
            self.qPB_zoom_in.setEnabled(True)
            self.qPB_zoom_out.setEnabled(False)
        elif mode == ZOOM_IN:
            self.qL_zoom.setEnabled(True)
            self.qPB_zoom_in.setEnabled(False)
            self.qPB_zoom_out.setEnabled(True)

    def imageEnhancement_setEnabled(self, flag):
        self.imageEnhancement_enabled = flag
        for item in (self.qL_image_enhancement,
                     self.qL_hue, self.qL_brightness, self.qL_saturation,
                     self.qL_contrast, self.qS_hue, self.qS_brightness,
                     self.qS_saturation, self.qS_contrast, self.qPB_reset_hue,
                     self.qPB_reset_brightness, self.qPB_reset_saturation,
                     self.qPB_reset_contrast):
            item.setEnabled(flag)

    def imageEnhancement_reset(self):
        self.qS_hue.setValue(-255)
        self.qS_brightness.setValue(0)
        self.qS_saturation.setValue(0)
        self.qS_contrast.setValue(0)

    def layers_setEnabled(self, flag):
        self.qTW_layers.setEnabled(flag)
        self.qPB_add.setEnabled(flag)

    def layers_add(self, flag, name=''):
        item_0 = QtWidgets.QTreeWidgetItem()
        QtWidgets.QTreeWidgetItem(item_0)
        QtWidgets.QTreeWidgetItem(item_0)
        if flag:
            self.current = sum(self.layers)
            self.qTW_layers.insertTopLevelItem(self.current, item_0)
        else:
            self.current = self.layers[1]
            self.qTW_layers.insertTopLevelItem(self.current, item_0)

        self.shapes.insert(self.current, [None, None])

        item_0.child(0).setBackground(1, QtGui.QColor(COL_BG_UNDEF))
        item_0.child(1).setBackground(1, QtGui.QColor(COL_BG_UNDEF))
        item_0.child(0).setForeground(1, QtGui.QColor(COL_FG_UNDEF))
        item_0.child(1).setForeground(1, QtGui.QColor(COL_FG_UNDEF))

        layer_name = f': {name}' if name != '' else ''
        item_0.setText(0, self.generate_layer_text(flag) + layer_name)
        item_0.child(0).setText(0, "Disc")
        item_0.child(0).setText(1, T_UNDEFINED)
        item_0.child(1).setText(0, "Cup")
        item_0.child(1).setText(1, T_UNDEFINED)

        button = QtWidgets.QRadioButton()
        self.qBG_layers.addButton(button)
        self.qTW_layers.setItemWidget(item_0, 1, button)

        button.click()
        item_0.setExpanded(True)
        self.qPB_remove.setEnabled(True)
        self.info_setEnabled(True)

        layer = self.get_layer_type(win.current)
        qimg = self.qImage
        if layer is False:
            if qimg.hasMask is False:
                qimg.load_automatic_layer()
            else:
                self.layers_activate_masks()
                self.info_update_all()

    def generate_layer_text(self, flag):
        if flag:
            self.layers[0] += 1
            return f'{T_MANUAL}{self.layers[0]}'
        else:
            self.layers[1] += 1
            return f'{T_AUTOMATIC}{self.layers[1]}'

    def set_mask_mode(self, flag):
        if flag:
            self.mask_setMode(MASK_MANUAL)
        else:
            self.mask_setMode(MASK_AUTOMATIC)

    def update_radio_buttons(self):
        for i in range(sum(self.layers)):
            item = self.qTW_layers.topLevelItem(i)
            self.qTW_layers.topLevelItem(i).setSelected(False)
            btn = self.qTW_layers.itemWidget(item, 1)
            if btn.isChecked():
                btn.setText(T_SELECTED)
                btn.setStyleSheet(f'background-color: {COL_BG_P_SEL};'
                                  f'color: {COL_FG_P_SEL};')
                item.setExpanded(True)
                item.setForeground(0, QtGui.QColor(COL_FG_P_SEL))
                item.setBackground(0, QtGui.QColor(COL_BG_P_SEL))
                item.setForeground(1, QtGui.QColor(COL_FG_P_SEL))
                item.setBackground(1, QtGui.QColor(COL_BG_P_SEL))
                item.child(0).setForeground(0, QtGui.QColor(COL_FG_C_SEL))
                item.child(0).setBackground(0, QtGui.QColor(COL_BG_C_SEL))
                item.child(1).setForeground(0, QtGui.QColor(COL_FG_C_SEL))
                item.child(1).setBackground(0, QtGui.QColor(COL_BG_C_SEL))
                flag = item.text(0)[0] == T_MANUAL[0]
                self.set_mask_mode(flag)
                self.current = i
            else:
                btn.setStyleSheet('')
                item.setData(0, QtCore.Qt.ForegroundRole, None)
                item.setData(0, QtCore.Qt.BackgroundRole, None)
                item.setData(1, QtCore.Qt.ForegroundRole, None)
                item.setData(1, QtCore.Qt.BackgroundRole, None)
                item.child(0).setData(0, QtCore.Qt.ForegroundRole, None)
                item.child(0).setData(0, QtCore.Qt.BackgroundRole, None)
                item.child(1).setData(0, QtCore.Qt.ForegroundRole, None)
                item.child(1).setData(0, QtCore.Qt.BackgroundRole, None)
                btn.setText(T_NOTSELECTED)

    def update_layer_info_color(self, item):
        status = item.text(1)[0] == T_UNDEFINED[0]
        if status:
            item.setForeground(1, QtGui.QColor(COL_FG_UNDEF))
            item.setBackground(1, QtGui.QColor(COL_BG_UNDEF))
        else:
            item.setForeground(1, QtGui.QColor(COL_FG_DEF))
            item.setBackground(1, QtGui.QColor(COL_BG_DEF))

    def update_layers_info(self):
        for i in range(sum(self.layers)):
            item = self.qTW_layers.topLevelItem(i)
            for j in range(2):
                self.update_layer_info_color(item.child(j))

    def get_layer_type(self, idx):
        text = self.qTW_layers.topLevelItem(idx).text(0)[0]
        return text == T_MANUAL[0]

    def strike_out_text(self, item):
        f = item.font()
        f.setStrikeOut(True)
        item.setFont(f)

    def create_disc_mask_menu(self):
        menu = QtWidgets.QMenu(self.qPB_disc,
                               triggered=self.add_disc_mask_triggered)
        act1 = menu.addAction("Circle")
        act1.setIcon(QtGui.QIcon("resources/menu_circle.png"))
        act2 = menu.addAction("Ellipse")
        act2.setEnabled(False)
        act2.setIcon(QtGui.QIcon("resources/menu_ellipse.png"))
        self.strike_out_text(act2)
        self.qPB_disc.setMenu(menu)

    def create_cup_mask_menu(self):
        menu = QtWidgets.QMenu(self.qPB_cup,
                               triggered=self.add_cup_mask_triggered)
        act1 = menu.addAction("Circle")
        act1.setIcon(QtGui.QIcon("resources/menu_circle.png"))
        act2 = menu.addAction("Ellipse")
        act2.setEnabled(False)
        act2.setIcon(QtGui.QIcon("resources/menu_ellipse.png"))
        self.strike_out_text(act2)
        self.qPB_cup.setMenu(menu)

    def mask_add_menu_trig(self, action, i):
        if action.text() == "Circle":
            self.set_layer_child(i, True)
            self.shapes[self.current][i] = Circle()
        elif action.text() == "Ellipse":
            self.set_layer_child(i, True)
            self.shapes[self.current][i] = Ellipse()

        if self.zoom_mode == ZOOM_OUT:
            self.qPB_zoom_in.click()
        self.qImage.respawn()

    @QtCore.pyqtSlot(QtWidgets.QAction)
    def add_disc_mask_triggered(self, action):
        self.mask_add_menu_trig(action, 0)

    @QtCore.pyqtSlot(QtWidgets.QAction)
    def add_cup_mask_triggered(self, action):
        self.mask_add_menu_trig(action, 1)

    def mask_update_buttons_text(self, idx):
        item = self.qTW_layers.topLevelItem(idx)

        disc = item.child(0).text(1)[0] == T_DEFINED[0]
        cup = item.child(1).text(1)[0] == T_DEFINED[0]
        self.mask_layers[0], self.mask_layers[1] = disc, cup
        disc_str = 'Remove' if disc else 'Add'
        cup_str = 'Remove' if cup else 'Add'
        self.mask_rename_buttons(disc=disc_str,
                                 cup=cup_str)
        layer_type = self.get_layer_type(self.current)
        if not disc:
            if layer_type:
                self.create_disc_mask_menu()
            else:
                self.mask_rename_buttons(disc=T_BTN_EMPTY)
                self.qPB_disc.setMenu(None)
            self.mask_disc_setEnabled(False)
        else:
            self.mask_disc_setEnabled(True)
            self.qPB_disc.setMenu(None)
            if layer_type is False:
                self.mask_rename_buttons(disc=T_BTN_EMPTY)
                self.qCB_disc_outline.setEnabled(False)

        if not cup:
            if layer_type:
                self.create_cup_mask_menu()
            else:
                self.mask_rename_buttons(cup=T_BTN_EMPTY)
                self.qPB_cup.setMenu(None)
            self.mask_cup_setEnabled(False)
        else:
            self.mask_cup_setEnabled(True)
            self.qPB_cup.setMenu(None)
            if layer_type is False:
                self.mask_rename_buttons(cup=T_BTN_EMPTY)
                self.qCB_cup_outline.setEnabled(False)

    def layers_activate_masks(self):
        self.set_layer_child(0, True)
        self.set_layer_child(1, True)

    def layer_changed(self, rbtn):
        if rbtn.isChecked():
            self.update_radio_buttons()
            idx = self.current
            self.mask_update_buttons_text(idx)
            self.qImage.repaint()
            self.info_update_all()

    def set_isnt(self, isnt):
        if isinstance(isnt, tuple):
            flag = sum(1 if i else 0 for i in isnt) >= 2
            col = COL_PASS if flag else COL_FAIL
            joined = '-'.join(['1' if i else '0' for i in isnt])
            result = 'PASS' if flag else 'FAIL'
            self.qL_isnt_val.setText(f'{result} ({joined})')
            self.qL_isnt_val.setStyleSheet(f'color: {col}')
        else:
            self.qL_isnt_val.setText(isnt)
            self.qL_isnt_val.setStyleSheet(f'color: {COL_FAIL}')

    def set_cdr(self, cdr):
        if isinstance(cdr, float):
            self.qL_cdr_val.setText(f'{cdr:.3f}')
            col = COL_PASS if cdr < CDR_THRES else COL_FAIL
            self.qL_cdr_val.setStyleSheet(f'color: {col}')
        elif isinstance(cdr, str):
            self.qL_cdr_val.setText(cdr)
            self.qL_cdr_val.setStyleSheet(f'color: {COL_FAIL}')

    def set_dtr(self, dtr):
        if isinstance(dtr, float):
            self.qL_dtr_val.setText(f'{dtr*100:.1f} %')
            col = COL_PASS if dtr <= 0.5 else COL_FAIL
            self.qL_dtr_val.setStyleSheet(f'color: {col}')
        elif isinstance(dtr, str):
            self.qL_dtr_val.setText(dtr)
            self.qL_dtr_val.setStyleSheet(f'color: {COL_FAIL}')

    def set_layer_child(self, num, flag):
        idx = self.current
        if idx != -1:
            para = (COL_FG_DEF, COL_BG_DEF, T_DEFINED) if flag else \
                   (COL_FG_UNDEF, COL_BG_UNDEF, T_UNDEFINED)
            item = self.qTW_layers.topLevelItem(idx)
            item.child(num).setForeground(1, QtGui.QColor(para[0]))
            item.child(num).setBackground(1, QtGui.QColor(para[1]))
            item.child(num).setText(1, para[2])
            item.setSelected(False)
            self.mask_update_buttons_text(idx)

    def reset_frames_style(self):
        for i in (self.qF1, self.qF2, self.qF3, self.qF4, self.qV):
            i.setStyleSheet('')

    def mw_refresh_frames(self):
        for i in (self.qF1, self.qF2, self.qF3, self.qF4, self.qV):
            i.setStyleSheet(f'QFrame {{background-color: {COL_FRAME};}}')

    def reset_all_sliders(self):
        self.qCB_disc_outline.setChecked(True)
        self.qCB_cup_outline.setChecked(True)
        self.qS_disc_alpha.setValue(90)
        self.qS_cup_alpha.setValue(100)

        self.qS_hue.setValue(-255)
        self.qS_brightness.setValue(0)
        self.qS_saturation.setValue(0)
        self.qS_contrast.setValue(0)

    def menu_new(self):
        if self.isChanged is True:
            qm = QtWidgets.QMessageBox
            ans = qm.question(self,
                              'New Project Confirmation',
                              '<p>This will start a new project. Do you want'
                              ' to save current project?</p>',
                              qm.Yes | qm.No | qm.Cancel)

            if ans == qm.Yes or ans == qm.No:
                if ans == qm.Yes:
                    self.menu_save()
                self.qImage.reset_all()
                self.qImage.refresh()
                self.qTW_layers.clear()
                self.reset_all_sliders()
                self.qPB_remove.setEnabled(False)
                self.create_main_variables()
                self.isChanged = False

    def menu_open(self):
        print('Menubar: Open')

    def menu_save(self):
        print('TODO: Save Project')

    def menu_exit(self):
        if self.isChanged is True:
            qm = QtWidgets.QMessageBox
            ans = qm.question(self,
                              'Exit Confirmation',
                              '<p>Do you want save current project before'
                              ' you exit the application?</p>',
                              qm.Yes | qm.No | qm.Cancel)

            if ans == qm.Yes or ans == qm.No:
                if ans == qm.Yes:
                    self.menu_save()
                sys.exit(app.exit())
                return False
            else:
                return True
        else:
            sys.exit(app.exit())
            return False

    def closeEvent(self, event):
        if self.menu_exit():
            event.ignore()
        else:
            event.accept()

    def menu_set_dark_mode(self, flag):
        self.actionDarkMode.setChecked(flag)

    def menu_dark_mode(self):
        flag = self.actionDarkMode.isChecked()
        win.set_dark_mode(flag, refresh=True)
        win.save_theme_flag()

    def menu_about(self):
        try:
            with open(ABOUT_MESSAGE, 'r') as file:
                about_message = file.read()
        except FileNotFoundError:
            about_message = 'ERROR! File not found!'

        QtWidgets.QMessageBox.about(self,
                                    "About Glaucoma Detection",
                                    about_message)

    def menu_about_qt(self):
        app.aboutQt()

    def menu_github(self):
        print('TODO: GitHub Link')


def load_models():
    ''' Load CDR model class '''
    from detection_rate_model import DetectionRateModel, normalize_sample_data
    # Create global variable for CDR model
    global cdr_model, normalize_data
    CDR_MODEL = 'models/detection_rate_model.h5'
    cdr_model = DetectionRateModel(CDR_MODEL)
    normalize_data = normalize_sample_data
    # Hide progress bar (loading)
    win.mw_end_wait_for()


if __name__ == "__main__":
    import sys

    # Used for debugging purposes
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    # Create QApplication & QMainWindow
    UI_FILE = 'glaucoma_detection.ui'
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(UI_FILE)
    win.show()

    # End execution on QApplication exit
    sys.exit(app.exec_())
