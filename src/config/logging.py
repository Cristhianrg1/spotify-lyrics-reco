import logging
import sys

def setup_logging():
    """
    Configures the root logger to output to stdout with a specific format.
    For Cloud Run, simple text lines are often enough as GCP captures them.
    For more advanced usage, we could use python-json-logger.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Format: [LEVEL] [Logger] Message
    formatter = logging.Formatter("[%(levelname)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplication if re-run
    if root.handlers:
        for h in root.handlers:
            root.removeHandler(h)
            
    root.addHandler(handler)
    
    # Set some noisy libraries to WARNING
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
