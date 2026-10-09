"""The Qt half of live guide mode: prompt bar, screen capture and highlight.

Everything that needs PySide6 is in this module, so the rest of scopepilot
imports and runs without it. Reach it through `live.load_overlay()`, which
tells the user how to install PySide6 when it is missing. The arithmetic that
turns the model's answer into positions on the screen is in `live.py`.

Like the rest of scopepilot this only looks and points. The highlight window
lets every click through to the software underneath; nothing here clicks or
types for the user.

macOS: capturing the screen needs the Screen Recording permission for the app
scopepilot is started from (Terminal, iTerm, an editor). Without it macOS hands
back the wallpaper with no windows on it, and the answer will be that the
control is not on this screen.
"""

from __future__ import annotations

import signal
import sys
from collections.abc import Callable
from functools import partial

from PIL import Image
from PySide6.QtCore import QObject, QRect, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QCursor,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QScreen,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scopepilot import live
from scopepilot.imaging import HIGHLIGHT
from scopepilot.locate import locate
from scopepilot.model import Backend
from scopepilot.profiles import Profile
from scopepilot.types import Box, Guidance

SETTLE_MS = 200
"""How long to wait after hiding our own windows before capturing, so they are
really gone from the screen and the model does not see them."""
BAR_WIDTH = 520
CARD_WIDTH = 340
RING_STROKE = 3
RING_RADIUS = 6

HINT = "Enter to ask. Esc clears the highlight; Esc again quits."
LOOKING = "Looking at the screen..."

_BAR_STYLE = """
#bar { background: #1c1c20; border: 1px solid #5a5a62; }
#bar QLineEdit { background: #2a2a30; color: white; border: 1px solid #5a5a62;
                 border-radius: 4px; padding: 6px; font-size: 14px; }
#bar QLineEdit:disabled { color: #9a9aa2; }
#bar QLabel { color: #c8c8d0; font-size: 12px; }
#bar QToolButton { color: #c8c8d0; border: none; font-size: 16px; padding: 0 4px; }
"""
_CARD_STYLE = """
#card { background: rgba(28, 28, 32, 240); border: 1px solid rgba(255, 255, 255, 70);
        border-radius: 10px; }
#card QLabel { color: white; font-size: 13px; }
#card QLabel#title { font-size: 14px; font-weight: bold; }
#card QLabel#confidence { color: #b4b4bc; font-size: 12px; }
"""


def to_pil(image: QImage) -> Image.Image:
    """Convert what Qt captured into the image type `locate()` takes."""
    if image.isNull():
        raise OSError("the screen could not be captured")
    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    # Qt pads each row to a multiple of four bytes, so the row length is passed on.
    return Image.frombytes(
        "RGB", (rgb.width(), rgb.height()), bytes(rgb.constBits()), "raw", "RGB", rgb.bytesPerLine()
    )


def grab(screen: QScreen) -> Image.Image:
    """Capture one whole screen, at whatever resolution Qt hands back.

    The size of the result is not assumed anywhere: `live.image_to_screen`
    works out the scale from it.
    """
    return to_pil(screen.grabWindow(0).toImage())


def _box(rect: QRect) -> Box:
    return Box(rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())


def _one_line(error: BaseException) -> str:
    """An error as it fits in the status line."""
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__


class PromptBar(QFrame):
    """The small always-on-top bar the question is typed into."""

    asked = Signal(str)
    escaped = Signal()
    close_clicked = Signal()

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("bar")
        self.setStyleSheet(_BAR_STYLE)
        self.setWindowTitle("scopepilot")

        self.field = QLineEdit()
        self.field.setPlaceholderText("Ask how to do something...")
        # Bound methods, here and below, not lambdas that mention `self`: Qt
        # keeps a lambda alive, the lambda would keep this window alive, and a
        # window that outlives the application object can crash the exit.
        self.field.returnPressed.connect(self._submit)
        self.close_button = QToolButton()
        self.close_button.setText("×")
        self.close_button.setToolTip("Quit")
        self.close_button.clicked.connect(self.close_clicked)
        self.status = QLabel(HINT)
        # A reason from the model or the API can be a long sentence. On one
        # line the bar would cut it off where the bar ends.
        self.status.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(self.field)
        row.addWidget(self.close_button)
        column = QVBoxLayout(self)
        column.addLayout(row)
        column.addWidget(self.status)
        self.setFixedWidth(BAR_WIDTH)
        self._fit()

    def _submit(self) -> None:
        self.asked.emit(self.field.text())

    def _fit(self) -> None:
        """Take the height the status line needs.

        The bottom edge stays where it is: the bar starts near the bottom of
        the screen, and growing downward would push it off.
        """
        was = self.geometry()
        height = self.heightForWidth(BAR_WIDTH)
        self.setGeometry(was.x(), was.y() + was.height() - height, BAR_WIDTH, height)

    def set_state(self, status: str, *, busy: bool) -> None:
        """Show `status`; while busy the field is greyed out so a second question waits."""
        self.status.setText(status)
        self._fit()
        self.field.setEnabled(not busy)
        if not busy:
            # Selected, so typing the next question replaces the last one.
            self.field.setFocus()
            self.field.selectAll()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # The text field passes Esc up to here, and so does the window itself
        # while the field is greyed out and nothing has the focus.
        if event.key() == Qt.Key.Key_Escape:
            self.escaped.emit()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # A frameless window has no title bar to drag. The bar decides which
        # screen is captured, so it has to be movable to the other monitor.
        if event.button() == Qt.MouseButton.LeftButton and self.windowHandle() is not None:
            self.windowHandle().startSystemMove()


class Card(QFrame):
    """The caption next to the highlight: what the control is and what to do."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet(_CARD_STYLE)
        column = QVBoxLayout(self)
        self.title, self.steps, self.note, self.confidence = (
            self._line(column, name) for name in ("title", "steps", "note", "confidence")
        )

    def _line(self, column: QVBoxLayout, name: str) -> QLabel:
        label = QLabel()
        label.setObjectName(name)
        # The words come from a model. Plain text keeps them from being read as markup.
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        column.addWidget(label)
        return label

    def write(self, caption: live.Caption, max_width: float) -> None:
        """Fill in the words and take the size they need at the card's width."""
        texts = (caption.title, "\n".join(caption.steps), caption.note, caption.confidence)
        for label, text in zip((self.title, self.steps, self.note, self.confidence), texts):
            label.setText(text)
            label.setVisible(bool(text))
        width = int(min(CARD_WIDTH, max_width))
        self.layout().invalidate()
        self.resize(width, self.heightForWidth(width))


class Overlay(QWidget):
    """A see-through window over one screen that draws the ring and the card.

    It never takes the focus and passes all mouse input to whatever is under
    it, so the user clicks the highlighted control as if the overlay were not
    there.
    """

    def __init__(self) -> None:
        super().__init__(
            None,
            # Tool keeps it out of the taskbar and the window switcher.
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # macOS hides tool windows when their app is not the active one, which
        # is exactly when the user is clicking in the microscope software.
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.card = Card(self)
        self.placement: live.Placement | None = None

    def point(
        self, guidance: Guidance, image_size: tuple[int, int], screen: QScreen, bar: Box
    ) -> Box:
        """Cover `screen` and draw the answer to a capture of it that was `image_size` pixels.

        `bar` is the prompt bar in desktop coordinates. Returns where the bar
        should be now, also in desktop coordinates: somewhere else only when it
        was sitting on the control or the card.
        """
        geometry = screen.geometry()
        area = live.Screen(geometry.x(), geometry.y(), geometry.width(), geometry.height())
        self.card.write(live.caption(guidance), area.width - 2 * live.MARGIN)
        card_size = (self.card.width(), self.card.height())
        self.placement = live.arrange(guidance.box, image_size, area, card_size, bar)
        self.card.move(round(self.placement.card.left), round(self.placement.card.top))
        self.setScreen(screen)
        self.setGeometry(geometry)
        self.show()
        self.update()
        return live.screen_to_desktop(self.placement.bar, area)

    def paintEvent(self, event: QPaintEvent) -> None:
        ring = self.placement.ring if self.placement is not None else None
        if ring is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outline = QRectF(ring.left, ring.top, ring.width, ring.height)
        # A pale halo under the coloured stroke keeps the ring visible on a
        # background that happens to be the same colour.
        halo = (QColor(255, 255, 255, 200), RING_STROKE + 4)
        for color, stroke in (halo, (QColor(*HIGHLIGHT), RING_STROKE)):
            painter.setPen(QPen(color, stroke))
            painter.drawRoundedRect(outline, RING_RADIUS, RING_RADIUS)


class _Lookup(QThread):
    """One `locate()` call on its own thread, so the prompt bar never freezes.

    A model call takes seconds to minutes. The result comes back to the GUI
    thread as a signal.
    """

    answered = Signal(object)
    failed = Signal(str)

    def __init__(self, job: Callable[[], Guidance]) -> None:
        super().__init__()
        self._job = job

    def run(self) -> None:
        try:
            guidance = self._job()
        except BaseException as error:
            # Whatever went wrong, the user has to hear about it and get the
            # bar back; an exception lost on this thread would leave it greyed
            # out. BaseException, because a SystemExit that gets out of here
            # ends the whole process with Qt's "destroyed while thread is still
            # running" abort.
            self.failed.emit(_one_line(error))
        else:
            self.answered.emit(guidance)


class Guide(QObject):
    """The prompt bar, the overlay, and the steps from a question to its answer."""

    answered = Signal(object)
    failed = Signal(str)
    closed = Signal()

    def __init__(
        self,
        profile: Profile,
        backend: Backend,
        *,
        refine: bool = True,
        capture: Callable[[QScreen], Image.Image] | None = None,
        settle_ms: int = SETTLE_MS,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._backend = backend
        self._refine = refine
        self._capture = capture or grab
        self._settle_ms = settle_ms
        self._busy = False
        self._closed = False
        self._bar_was_up = False
        self._shot: tuple[tuple[int, int], QScreen] | None = None
        self._lookup: _Lookup | None = None

        self.bar = PromptBar()
        self.overlay = Overlay()
        self.bar.asked.connect(self.ask)
        self.bar.escaped.connect(self.escape)
        self.bar.close_clicked.connect(self.close)
        self._place_bar()

    @property
    def busy(self) -> bool:
        """Whether a question is waiting for its answer."""
        return self._busy

    def _place_bar(self) -> None:
        """Start on the screen the mouse is on, clear of the taskbar or Dock."""
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        x, y = live.bar_position(
            (self.bar.width(), self.bar.height()),
            live.Screen(area.x(), area.y(), area.width(), area.height()),
        )
        self.bar.move(round(x), round(y))

    def start(self) -> None:
        """Show the prompt bar and wait for questions."""
        self.bar.show()
        self.bar.activateWindow()
        self.bar.field.setFocus()

    def ask(self, question: str) -> None:
        """Answer `question` about whatever is on the screen the bar is on."""
        question = question.strip()
        if not question or self._busy:
            return
        self._busy = True
        self._bar_was_up = self.bar.isVisible()
        self.bar.hide()
        self.overlay.hide()
        QTimer.singleShot(self._settle_ms, partial(self._look, question))

    def _look(self, question: str) -> None:
        """Capture the screen, then hand the slow part to another thread."""
        if self._closed:
            return
        # The screen under the bar's centre. Not `self.bar.screen()`: PySide
        # ties the object that returns to the bar, and it goes dead with it.
        centre = self.bar.geometry().center()
        screen = QGuiApplication.screenAt(centre) or QGuiApplication.primaryScreen()
        try:
            image = self._capture(screen)
        except Exception as error:
            # As in _Lookup.run: any failure must end with the bar back.
            self._fail(_one_line(error))
            return
        self._shot = (image.size, screen)
        self._bar_back(LOOKING, busy=True)
        if self._lookup is not None:
            # The last answer has arrived, but its thread may still be winding down.
            self._lookup.wait()
        self._lookup = _Lookup(
            partial(locate, image, question, self._profile, self._backend, refine=self._refine)
        )
        self._lookup.answered.connect(self._show)
        self._lookup.failed.connect(self._fail)
        self._lookup.start()

    @Slot(object)
    def _show(self, guidance: Guidance) -> None:
        if self._closed:
            return
        image_size, screen = self._shot
        try:
            bar = self.overlay.point(guidance, image_size, screen, _box(self.bar.geometry()))
            self.bar.move(round(bar.left), round(bar.top))
        except Exception as error:
            # As in _Lookup.run. Qt only prints an exception raised in a slot
            # and carries on, so without this the bar would stay greyed out,
            # and `--ask` would wait for ever with nothing on screen.
            self.overlay.hide()
            self._fail(_one_line(error))
            return
        title = live.caption(guidance).title
        self._bar_back(title if guidance.box is None else f"Highlighted: {title}", busy=False)
        self._busy = False
        self.answered.emit(guidance)

    @Slot(str)
    def _fail(self, message: str) -> None:
        if self._closed:
            return
        self._bar_back(message, busy=False)
        self._busy = False
        self.failed.emit(message)

    def _bar_back(self, status: str, *, busy: bool) -> None:
        """Update the bar and, if it was showing when the question came, show it again."""
        self.bar.set_state(status, busy=busy)
        if self._bar_was_up:
            self.bar.show()
            # Above the overlay, so the card can never cover the field.
            self.bar.raise_()

    def escape(self) -> None:
        """Esc: clear the highlight if there is one, otherwise quit."""
        if self.overlay.isVisible():
            self.overlay.hide()
        else:
            self.close()

    def close(self) -> None:
        """Take both windows down for good. An answer that arrives later is dropped."""
        self._closed = True
        self.bar.hide()
        self.overlay.hide()
        self.closed.emit()

    def shutdown(self) -> None:
        """Wait for a model call that is still running.

        Its thread cannot be interrupted, and Qt aborts the process if a thread
        object is destroyed while running.
        """
        if self._lookup is not None:
            self._lookup.wait()


def run(
    profile: Profile,
    backend: Backend,
    *,
    question: str | None = None,
    seconds: float = 15.0,
    refine: bool = True,
) -> int:
    """Run guide mode and return the process exit code.

    Without `question`, shows the prompt bar until the user quits. With one,
    answers it, leaves the highlight up for `seconds`, and exits.
    """
    if question is not None and not question.strip():
        raise ValueError("--ask needs a question")
    if seconds < 0:
        raise ValueError("--seconds cannot be negative")

    app = QApplication.instance() or QApplication(sys.argv[:1])
    # The bar is hidden for every capture; that must not count as closing the app.
    app.setQuitOnLastWindowClosed(False)
    guide = Guide(profile, backend, refine=refine)
    guide.closed.connect(app.quit)
    errors: list[str] = []
    if question is None:
        guide.start()
    else:
        linger = QTimer()
        linger.setSingleShot(True)
        linger.setInterval(round(seconds * 1000))
        linger.timeout.connect(guide.close)
        guide.answered.connect(lambda _: linger.start())
        guide.failed.connect(errors.append)
        guide.failed.connect(guide.close)
        QTimer.singleShot(0, partial(guide.ask, question))

    # While Qt's event loop runs, or waits for a thread, Python never gets to
    # raise KeyboardInterrupt. The default handler lets Ctrl+C in the terminal
    # stop a guide whose bar is hidden.
    previous = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        app.exec()
        if guide.busy:
            print("scopepilot: waiting for the model to answer; Ctrl+C stops now", file=sys.stderr)
        guide.shutdown()
    finally:
        signal.signal(signal.SIGINT, previous)
    if errors:
        print(f"scopepilot: {errors[0]}", file=sys.stderr)
        return 1
    return 0
