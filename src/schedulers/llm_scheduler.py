# llm_scheduler.py
'''
@desc
    This module implements the LLM-based scheduler.
    It connects to a Generative AI API (like Gemini) to make
    intelligent scheduling decisions based on the current
    simulation state.
    
    This module is designed to be swapped in to replace the
    BaselineScheduler.
'''
import os
import re
import google.generativeai as genai

# Import type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.nodes.satellite_mec import SatelliteMEC, Task
    from src.simlogging.ilogger import ILogger
    from src.utils import Time

# --- IMPORTANT: SET YOUR API KEY ---
# Set this in your system's environment variables
# (e.g., export GEMINI_API_KEY="your_key_here")
API_KEY = os.environ.get("GEMINI_API_KEY")

class LLMScheduler:
    '''
    @desc
        A scheduler that uses a Large Language Model (LLM)
        to make real-time scheduling decisions.
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
        
        if not API_KEY:
            self.logger.write_Log("GEMINI_API_KEY environment variable not set. LLMScheduler will not work.", "LOGERROR", Time())
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
            
        # Configure the Gemini client
        genai.configure(api_key=API_KEY)
        
        # --- Configure the Generative Model ---
        # You can adjust generation_config as needed for your project
        generation_config = {
            "temperature": 0.2, # Low temperature for more deterministic, less "creative" answers
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 50, # We only need a few tokens for the ID
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-pro", # Use a fast, capable model
            generation_config=generation_config
        )
        
        self.logger.write_Log("LLMScheduler initialized and Gemini client configured.", "LOGINFO", Time())

    def _state_to_prompt(self, task: 'Task', current_time: 'Time') -> str:
        '''
        @desc
            (Step 3.1) Converts the current simulation state into a 
            natural language prompt for the LLM.
        '''
        
        prompt = "You are an expert resource allocator for a LEO satellite network.\n"
        prompt += f"Your goal is to minimize latency by selecting the best satellite to process a new task.\n"
        prompt += f"Current simulation time: {current_time.to_str()}.\n\n"
        prompt += "--- NEW TASK ---\n"
        prompt += f"Task ID: {task.id}\n"
        prompt += f"Computational Load: {task.mips_required} MIPS\n\n"
        
        prompt += "--- SATELLITE STATUS ---\n"
        for sat in self.satellites:
            load_percent = sat.get_current_load_percentage()
            queue_length = len(sat.cpu_resource.queue)
            
            prompt += f"Satellite ID: {sat.nodeID}\n"
            prompt += f"  - CPU Capacity: {sat.cpu_processing_rate} MIPS/sec\n"
            prompt += f"  - Current CPU Load: {load_percent:.1f}%\n"
            prompt += f"  - Tasks in Queue: {queue_length}\n"
        
        prompt += "\n--- DECISION ---\n"
        prompt += "Based on this data, which satellite (ID) should process the new task?\n"
        prompt += "Respond with the numerical Satellite ID only."
        
        return prompt

    def _call_gemini_api(self, prompt: str) -> str:
        '''
        @desc
            (Step 3.2) Sends the prompt to the Gemini API and returns the
            raw text response.
        '''
        try:
            self.logger.write_Log(f"Sending prompt to Gemini API...", "LOGDEBUG", Time())
            # self.logger.write_Log(f"PROMPT:\n{prompt}", "LOGDEBUG", Time()) # Uncomment for full prompt logging
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            self.logger.write_Log(f"LLM raw response: '{response_text}'", "LOGDEBUG", Time())
            return response_text
            
        except Exception as e:
            self.logger.write_Log(f"Gemini API call failed: {e}", "LOGERROR", Time())
            return None  # Return None to signal a failure

    def _response_to_action(self, response_text: str) -> 'SatelliteMEC':
        '''
        @desc
            (Step 3.3) Parses the LLM's text response and converts it
            into a valid simulation action (a SatelliteMEC object).
            Includes fallback logic.
        '''
        
        # 1. Handle API Failure
        if response_text is None:
            self.logger.write_Log("API response was None. Using fallback logic.", "LOGWARN", Time())
            return self._fallback_logic()

        # 2. Parse the text response
        try:
            # Use regex to find the first sequence of digits in the response
            match = re.search(r'\d+', response_text)
            
            if match:
                chosen_id = int(match.group(0))
                
                # Find the satellite object that matches the chosen ID
                for sat in self.satellites:
                    if sat.nodeID == chosen_id:
                        self.logger.write_Log(f"LLM parsed action: Assign to {chosen_id}", "LOGINFO", Time())
                        return sat
                
                # If ID is valid number but doesn't exist
                self.logger.write_Log(f"LLM returned non-existent satellite ID: {chosen_id}. Using fallback.", "LOGWARN", Time())
                return self._fallback_logic()
            
            else:
                # If no number was found in the response
                self.logger.write_Log(f"LLM response '{response_text}' contained no valid ID. Using fallback.", "LOGWARN", Time())
                return self._fallback_logic()

        except Exception as e:
            self.logger.write_Log(f"Error parsing LLM response '{response_text}': {e}. Using fallback.", "LOGWARN", Time())
            return self._fallback_logic()

    def _fallback_logic(self) -> 'SatelliteMEC':
        '''
        @desc
            A robust fallback in case the LLM API fails or
            returns an invalid response. Defaults to the baseline logic.
        '''
        self.logger.write_Log("Executing fallback: Assigning to shortest queue.", "LOGWARN", Time())
        chosen_satellite = min(self.satellites, key=lambda sat: len(sat.cpu_resource.queue))
        return chosen_satellite

    def schedule_task(self, task: 'Task', current_time: 'Time') -> 'SatelliteMEC':
        '''
        @desc
            The main decision-making method. Orchestrates the
            prompt -> API -> parse workflow.
        @param[in] task
            The task object to be scheduled.
        @param[in] current_time
            The current simulation timestamp.
        @return
            The chosen SatelliteMEC object to process the task.
        '''
        
        # 1. Create prompt from current state
        prompt = self._state_to_prompt(task, current_time)
        
        # 2. Call LLM
        response_text = self._call_gemini_api(prompt)
        
        # 3. Parse response and return the chosen satellite
        selected_satellite = self._response_to_action(response_text)
        
        return selected_satellite