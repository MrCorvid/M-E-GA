import datetime
import json
import os

# Import LoggingManager
from ..networkCommon.logging_manager import LoggingManager, VERBOSE_LEVEL_NUM # Corrected import path

class GA_Logger:
    def __init__(self, experiment_name, log_directory="logs"): # Updated default log directory
        """
        Initialize the GA_Logger instance.

        :param experiment_name: The name of the experiment, used for generating log filenames.
        :param log_directory: The directory where log files will be stored. Defaults to "logs".
        """
        self.experiment_name = experiment_name
        self.log_directory = log_directory
        # Initialize logger instance using LoggingManager
        self.logger = LoggingManager.get_logger("MEGA.GALogger") # Added logger instance

        if not os.path.exists(self.log_directory):
            try:
                os.makedirs(self.log_directory)
            except OSError as e:
                 # Use logger for critical init errors if possible, fallback to print
                 self.logger.critical(f"Could not create GA log directory {self.log_directory}: {e}", exc_info=True)
                 # Fallback if logger failed
                 print(f"CRITICAL ERROR (GA_Logger): Could not create log directory {self.log_directory}: {e}", flush=True)
                 # Decide how to proceed - maybe disable saving? For now, continue but saving will fail.

        self.events = []  # List to store all logged events.
        self.subscribers = []  # List of subscriber callback functions for real-time event notifications.
        # Use a fixed filename in the local folder.
        self.filename = os.path.join(self.log_directory, f"{self.experiment_name}_compiled_log.json")
        self.logger.debug(f"GA_Logger initialized. Log file: {self.filename}")

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def log_event(self, event_type, details):
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self.events.append(event)
        # Notify subscribers.
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                # Use logger instead of print
                self.logger.error(f"Error in subscriber callback: {e}", exc_info=True)

    def save(self, filename=None):
        """
        Append all collected events to a fixed JSON log file.
        Each event is written as a separate JSON object (one per line).
        After saving, the in-memory events are cleared.
        """
        if filename is None:
            filename = self.filename

        # Check directory existence again before writing
        if not os.path.exists(self.log_directory):
             self.logger.error(f"Log directory '{self.log_directory}' does not exist. Cannot save GA log.")
             self.events = [] # Clear events even if save fails to prevent memory leak
             return

        try:
            with open(filename, "a") as f:
                for event in self.events:
                    f.write(json.dumps(event) + "\n")
            self.logger.debug(f"Saved {len(self.events)} events to {filename}")
            self.events = []  # Clear events after saving.
        except Exception as e:
             self.logger.error(f"Failed to save GA events to {filename}: {e}", exc_info=True)
             # Decide whether to clear events on failure - clearing prevents potential memory leak
             self.events = []
