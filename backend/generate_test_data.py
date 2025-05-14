import json
import os
import random
from datetime import datetime, timedelta

# Load test data
with open('test_novelty_data.json', 'r') as f:
    test_data = json.load(f)

# Create novelty_counts.json
with open('novelty_counts.json', 'w') as f:
    json.dump(test_data['novelty_counts'], f, indent=2)
    print("Created novelty_counts.json")

# Create learning_log_metasploit.json
with open('learning_log_metasploit.json', 'w') as f:
    json.dump(test_data['learning_log'], f, indent=2)
    print("Created learning_log_metasploit.json")

# Create curiosity.log
with open('curiosity.log', 'w') as f:
    for log_line in test_data['curiosity_log']:
        f.write(log_line + '\n')
    print("Created curiosity.log")

print("Test data generation complete. You can now test the novelty insights page.")
