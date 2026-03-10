# baseline_scheduler.py
'''
@desc
    This module implements a simple, non-AI baseline scheduler.
    It uses a common heuristic (shortest processing queue) to make
    scheduling decisions.
    
    This serves as the benchmark to compare against the LLMScheduler.
'''

# Import type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.nodes.satellite_mec import SatelliteMEC, Task
    from src.simlogging.ilogger import ILogger
    from src.utils import Time

class BaselineScheduler:
    '''
    @desc
        A simple scheduler that assigns new tasks to the 
        satellite with the shortest active processing queue.
    '''
    
    def __init__(self, satellites: 'list[SatelliteMEC]', logger: 'ILogger'):
        '''
        @param[in] satellites
            A list of all SatelliteMEC node objects in the simulation.
        @param[in] logger
            The simulation logger instance.
        '''
        self.satellites = satellites
        self.logger = logger
        self.logger.write_Log("BaselineScheduler initialized.", "LOGINFO", Time())

    def schedule_task(self, task: 'Task', current_time: 'Time') -> 'SatelliteMEC':
        '''
        @desc
            The main decision-making method.
        @param[in] task
            The task object to be scheduled.
        @param[in] current_time
            The current simulation timestamp.
        @return
            The chosen SatelliteMEC object to process the task.
        '''
        
        if not self.satellites:
            self.logger.write_Log("Scheduler has no satellites to schedule to!", "LOGERROR", current_time)
            return None

        # --- HEURISTIC LOGIC ---
        # Find the satellite with the minimum number of tasks currently in its CPU queue.
        # This is a "shortest queue" or "least busy" strategy.
        try:
            chosen_satellite = min(self.satellites, key=lambda sat: len(sat.cpu_resource.queue))
        except Exception as e:
            self.logger.write_Log(f"Error finding min queue satellite: {e}. Defaulting to first sat.", "LOGWARN", current_time)
            chosen_satellite = self.satellites[0]
        # ---------------------

        self.logger.write_Log(f"BaselineScheduler: Assigning Task {task.id} to Satellite {chosen_satellite.nodeID}", "LOGINFO", current_time)
        return chosen_satellite