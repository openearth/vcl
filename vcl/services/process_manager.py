"""Process manager for coordinating VCL system processes.

Handles launching display windows and input handlers in the process pool,
with configuration injection and lifecycle management.
"""

import concurrent.futures
import logging
import os
import threading
from pathlib import Path
from typing import Optional, List, Callable, Any

logger = logging.getLogger(__name__)


class ProcessManager:
    """Coordinates launching and managing VCL processes.
    
    Manages display windows, input handlers, and tracking modules through
    a process pool executor. Provides a clean interface for starting/stopping
    components with proper error handling and lifecycle management.
    
    Example:
        manager = ProcessManager(max_workers=10)
        manager.start_museum_input(inactivity_timeout=120)
        manager.start_display_map(data_path, museum_mode=True)
        manager.start_all()  # Blocks until processes complete
    """
    
    def __init__(self, max_workers: int = 10):
        """Initialize process manager.
        
        Args:
            max_workers: Maximum number of worker processes in pool
        """
        self.max_workers = max_workers
        self.executor = None
        self.submitted_tasks: List[concurrent.futures.Future] = []
        self._watch_parent_thread = None
    
    def start(self) -> None:
        """Start the process pool executor.
        
        Initializes the process pool and starts parent process monitoring.
        """
        if self.executor is not None:
            logger.warning("ProcessManager already started")
            return
        
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker_process,
            initargs=(os.getpid(),),
        )
        logger.info(f"ProcessManager started with {self.max_workers} workers")
    
    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """Submit a function to be run in a worker process.
        
        Args:
            fn: Callable to execute
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn
            
        Returns:
            concurrent.futures.Future: Future for the submitted task
        """
        if self.executor is None:
            raise RuntimeError("ProcessManager not started. Call start() first.")
        
        future = self.executor.submit(fn, *args, **kwargs)
        self.submitted_tasks.append(future)
        return future
    
    def wait_all(self) -> None:
        """Block until all submitted tasks complete."""
        if not self.submitted_tasks:
            logger.warning("No tasks submitted")
            return
        
        logger.info(f"Waiting for {len(self.submitted_tasks)} tasks to complete...")
        concurrent.futures.wait(self.submitted_tasks)
        logger.info("All tasks completed")
    
    def shutdown(self) -> None:
        """Shutdown the process pool gracefully."""
        if self.executor is None:
            return
        
        logger.info("Shutting down ProcessManager...")
        self.executor.shutdown(wait=True)
        self.executor = None
        logger.info("ProcessManager shutdown complete")


def _init_worker_process(parent_pid: int) -> None:
    """Initializer for worker processes in the process pool.
    
    Sets up each worker to monitor and terminate if parent process dies.
    This ensures clean cleanup of child processes if parent crashes.
    
    Args:
        parent_pid: Parent process ID to monitor
    """
    # Start a daemon thread that monitors parent process
    def watch_parent():
        """Monitor parent process and terminate worker if parent dies."""
        import time
        import os
        import psutil
        
        while True:
            try:
                # Check if parent is still alive
                parent = psutil.Process(parent_pid)
                if not parent.is_running():
                    logger.warning(f"Parent process {parent_pid} died, terminating worker")
                    os._exit(1)
            except (psutil.NoSuchProcess, ProcessLookupError):
                logger.warning(f"Parent process {parent_pid} not found, terminating worker")
                os._exit(1)
            
            time.sleep(1)
    
    # Try to import psutil, if not available just skip parent monitoring
    try:
        import psutil
        thread = threading.Thread(target=watch_parent, daemon=True)
        thread.start()
    except ImportError:
        logger.debug("psutil not installed, parent process monitoring disabled")
