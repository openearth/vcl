class ActionManager:
    def __init__(self):
        self.actions = {}

    def register_action(self, uid_pattern, callback):
        """
        Register a callback function for a specific UID.
        uid_pattern: The string ID to match (exact match for now).
        callback: Function to call. Should accept (uid, context).
        """
        self.actions[uid_pattern] = callback

    def execute_action(self, uid, context):
        """
        Execute the action associated with the UID.
        uid: The detected ID string.
        context: Dictionary containing additional info (location, etc.)
        """
        if uid in self.actions:
            # print(f"Action triggered for UID: {uid}")
            self.actions[uid](uid, context)
        else:
            # Optional: Default action or simple logging
            # print(f"No action registered for UID: {uid}")
            pass
