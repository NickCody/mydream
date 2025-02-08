import uvicorn
from app.api import app
import logging
import sys
import time

# Define a safe logging handler that ignores BrokenPipeError.
class SafeStreamHandler(logging.StreamHandler):
    def flush(self):
        try:
            super().flush()
        except BrokenPipeError:
            pass

# Replace any existing StreamHandler on the uvicorn access logger.
access_logger = logging.getLogger("uvicorn.access")
for handler in list(access_logger.handlers):
    if isinstance(handler, logging.StreamHandler):
        access_logger.removeHandler(handler)
        access_logger.addHandler(SafeStreamHandler(sys.stdout))

def main():
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug", access_log=True)
    except KeyboardInterrupt:
        print("Server interrupted by user. Exiting gracefully in 5 seconds...")
        time.sleep(5)
        sys.exit(0)
    except Exception as e:
        logging.exception("An error occurred while running the server:")
        print("Exiting in 5 seconds...")
        time.sleep(5)
        sys.exit(1)

if __name__ == "__main__":
    main()