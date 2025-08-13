import time
import traceback

import schedule
from datetime import datetime


class Runnable:
    """
    An interface for tasks that can be run periodically.
    """

    def tick(self):
        """
        This method contains the logic that should be executed on each run.
        """
        raise NotImplementedError


class Runner:
    """
    A general-purpose task runner that executes tasks based on a schedule.
    """

    def __init__(self):
        self.tasks = []
        print("Runner initialized.")

    def add_task(self, runnable: Runnable, interval_minutes: int = None, at_time: str = None,
                 align_to_clock: bool = False):
        """
        Adds a task to the scheduler.

        :param runnable: An object that implements the Runnable interface.
        :param interval_minutes: The interval in minutes to run the task.
        :param at_time: The specific time of day to run the task (e.g., "23:55").
        :param align_to_clock: If True, aligns the interval task to the clock
                               (e.g., for a 5-minute interval, runs at HH:00, HH:05, HH:10).
        """
        def wrapped_tick():
            try:
                runnable.tick()
            except Exception as e:
                traceback.print_exc()

        job_func = wrapped_tick

        if interval_minutes:
            if align_to_clock:
                print(f"Scheduling task to run every {interval_minutes} minutes, aligned to the clock.")

                # This stateful class ensures the job runs only once per aligned minute.
                class AlignedJob:
                    def __init__(self):
                        self.last_run_minute = -1

                    def wrapper(self):
                        now = datetime.now()
                        current_minute = now.minute

                        # Check if we are in a new minute and if it's an aligned interval
                        if current_minute != self.last_run_minute and current_minute % interval_minutes == 0:
                            self.last_run_minute = current_minute
                            job_func()

                aligned_job = AlignedJob()
                # Schedule the check to run every second to reliably catch the start of the minute.
                schedule.every().second.do(aligned_job.wrapper)
            else:
                # Original logic for non-aligned tasks.
                schedule.every(interval_minutes).minutes.do(job_func)
                print(f"Scheduled task to run every {interval_minutes} minutes.")
        elif at_time:
            # Schedule a task to run once a day at a specific time.
            schedule.every().day.at(at_time).do(job_func)
            print(f"Scheduled task to run daily at {at_time}.")
        else:
            print("Error: Task must have either an interval or a specific run time.")

    def loop(self):
        """
        The main loop that continuously checks for and runs pending tasks.
        """
        print("Runner loop started. Waiting for scheduled tasks...")
        while True:
            schedule.run_pending()
            time.sleep(0.5)  # Sleep briefly to be CPU-friendly
