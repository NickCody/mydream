import sys
import argparse
from PyQt5 import QtWidgets
from gui.main_window import MainWindow

# Global variable to store the server URL
SERVER_URL = "http://localhost:8000"  # Default value

def parse_args():
    """ Parses command-line arguments to get the server URL. """
    global SERVER_URL

    parser = argparse.ArgumentParser(description="Start the RumpleMyDream GUI")
    parser.add_argument("--server-url", type=str, default=SERVER_URL, help="Set the server URL (default: http://localhost:8000)")

    args = parser.parse_args()
    SERVER_URL = args.server_url  # Update the global variable
    print(f"🌐 Server URL set to: {SERVER_URL}")
    
def main():
    """ Main function to start the GUI. """
    parse_args()  # Parse command-line arguments

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(SERVER_URL)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()