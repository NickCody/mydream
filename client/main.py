import sys
import argparse
from PyQt5 import QtWidgets
from gui.main_window import MainWindow

# Global variable to store the server URL
SERVER_URL = "http://localhost:8000"  # Default value

def parse_args():
    """Parses command-line arguments to get the server URL, width, and height."""
    global SERVER_URL

    # Disable default help to free up -h for height
    parser = argparse.ArgumentParser(
        description="Start the RumpleMyDream GUI", 
        add_help=False
    )
    
    # Custom help flag
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    
    parser.add_argument("--server-url", type=str, default=SERVER_URL,
                        help="Set the server URL (default: http://localhost:8000)")
    parser.add_argument("-w", "--width", type=int, default=640,
                        help="Set the width of the window (default: 640)")
    parser.add_argument("-h", "--height", type=int, default=512,
                        help="Set the height of the window (default: 512)")

    args = parser.parse_args()
    SERVER_URL = args.server_url  # Update the global variable

    print(f"🌐 Server URL set to: {SERVER_URL}")
    print(f"🖥 Window width: {args.width}")
    print(f"🖥 Window height: {args.height}")
    
    return args
    
def main():
    """ Main function to start the GUI. """
    args = parse_args()  # Parse command-line arguments

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)  # macOS may remove the tray if the last window is closed
    window = MainWindow(args, app)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()