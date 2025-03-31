import datetime
import json
import os

class GA_Logger:
    def __init__(self, experiment_name, log_directory="logs_and_log_tools"):
        """
        Initialize the GA_Logger instance.
        
        :param experiment_name: The name of the experiment, used for generating log filenames.
        :param log_directory: The directory where log files will be stored. Defaults to "logs_and_log_tools".
        """
        self.experiment_name = experiment_name
        self.log_directory = log_directory  # Use the local folder.
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
        self.events = []  # List to store all logged events.
        self.subscribers = []  # List of subscriber callback functions for real-time event notifications.
        # Use a fixed filename in the local folder.
        self.filename = os.path.join(self.log_directory, f"{self.experiment_name}_compiled_log.json")

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
                print("Error in subscriber callback:", e)

    def save(self, filename=None):
        """
        Append all collected events to a fixed JSON log file.
        Each event is written as a separate JSON object (one per line).
        After saving, the in-memory events are cleared.
        """
        if filename is None:
            filename = self.filename

        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        with open(filename, "a") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")
        self.events = []  # Clear events after saving.
