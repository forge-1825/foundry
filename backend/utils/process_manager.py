import threading
import queue
import re
import time
from typing import Dict, Optional

class ProcessManager:
    def __init__(self):
        self.processes: Dict[str, dict] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def add_process(self, process, step_id: str) -> str:
        with self._lock:
            process_id = str(self._next_id)
            self._next_id += 1

        def monitor_output(proc, out_queue: queue.Queue):
            progress_pattern = re.compile(r'Progress: (\d+)%')
            epoch_pattern = re.compile(r'Epoch (\d+)/(\d+)')
            
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                
                if line:
                    # Check for progress updates
                    progress_match = progress_pattern.search(line)
                    if progress_match:
                        progress = int(progress_match.group(1))
                        out_queue.put(('progress', progress))
                    
                    # Check for epoch updates
                    epoch_match = epoch_pattern.search(line)
                    if epoch_match:
                        current, total = map(int, epoch_match.groups())
                        out_queue.put(('epoch', (current, total)))
                    
                    # Store output
                    out_queue.put(('output', line.strip()))

        output_queue = queue.Queue()
        monitor_thread = threading.Thread(
            target=monitor_output,
            args=(process, output_queue),
            daemon=True
        )
        monitor_thread.start()

        self.processes[process_id] = {
            'process': process,
            'step_id': step_id,
            'output_queue': output_queue,
            'monitor_thread': monitor_thread,
            'start_time': time.time(),
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'status': 'running',
            'output_lines': []
        }

        return process_id

    def get_process_status(self, process_id: str) -> Optional[dict]:
        process_info = self.processes.get(process_id)
        if not process_info:
            return None

        # Process all queued output
        while True:
            try:
                msg_type, data = process_info['output_queue'].get_nowait()
                if msg_type == 'progress':
                    process_info['progress'] = data
                elif msg_type == 'epoch':
                    process_info['current_epoch'], process_info['total_epochs'] = data
                elif msg_type == 'output':
                    process_info['output_lines'].append(data)
            except queue.Empty:
                break

        # Check if process has finished
        if process_info['process'].poll() is not None:
            process_info['status'] = 'completed' if process_info['process'].returncode == 0 else 'failed'

        return {
            'status': process_info['status'],
            'progress': process_info['progress'],
            'current_epoch': process_info['current_epoch'],
            'total_epochs': process_info['total_epochs'],
            'output': process_info['output_lines'][-100:],  # Last 100 lines
            'runtime': time.time() - process_info['start_time']
        }

    def stop_process(self, process_id: str) -> bool:
        process_info = self.processes.get(process_id)
        if not process_info:
            return False

        process_info['process'].terminate()
        process_info['status'] = 'stopped'
        return True
