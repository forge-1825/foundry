from flask import Blueprint, jsonify, request
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

novelty_bp = Blueprint('novelty', __name__)

# Constants
CURIOSITY_LOG_PATH = os.environ.get("CURIOSITY_LOG_PATH", "curiosity.log")
NOVELTY_COUNTS_PATH = os.environ.get("NOVELTY_COUNTS_PATH", "novelty_counts.json")
NOVELTY_TOPICS_PATH = os.environ.get("NOVELTY_TOPICS_PATH", "novelty_counts_topics.json")
LEARNING_LOG_PATH = os.environ.get("LEARNING_LOG_PATH", "learning_log_metasploit.json")

def read_curiosity_log(max_lines=100):
    """Read the curiosity log file and return the most recent lines."""
    try:
        if not os.path.exists(CURIOSITY_LOG_PATH):
            return []

        with open(CURIOSITY_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Return the most recent lines
        return [line.strip() for line in lines[-max_lines:]]
    except Exception as e:
        logger.error(f"Error reading curiosity log: {e}")
        return []

def read_novelty_counts():
    """Read the novelty counts JSON file."""
    try:
        if not os.path.exists(NOVELTY_COUNTS_PATH):
            return {}

        with open(NOVELTY_COUNTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading novelty counts: {e}")
        return {}

def read_topic_novelty_counts():
    """Read the topic-specific novelty counts JSON file."""
    try:
        if not os.path.exists(NOVELTY_TOPICS_PATH):
            return {}

        with open(NOVELTY_TOPICS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading topic novelty counts: {e}")
        return {}

def read_learning_log():
    """Read the learning log JSON file."""
    try:
        if not os.path.exists(LEARNING_LOG_PATH):
            return []

        with open(LEARNING_LOG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading learning log: {e}")
        return []

def parse_novelty_timeline():
    """Parse the learning log to create a timeline of novelty scores."""
    learning_log = read_learning_log()
    timeline = []

    for entry in learning_log:
        if 'timestamp' in entry and 'novelty_score' in entry:
            timeline.append({
                'timestamp': entry['timestamp'],
                'novelty_score': entry['novelty_score'],
                'command': entry.get('command', ''),
                'success': entry.get('success', True)
            })

    return timeline

def get_most_novel_states(limit=10, topic=None):
    """Get the most novel states based on visit counts."""
    if topic:
        # Get topic-specific novelty counts
        topic_counts = read_topic_novelty_counts()
        if topic not in topic_counts:
            return []
        novelty_counts = topic_counts[topic]
    else:
        # Get global novelty counts
        novelty_counts = read_novelty_counts()

    # Convert to list of (state, count) tuples
    state_counts = [(state, count) for state, count in novelty_counts.items()]

    # Sort by count (ascending) to get the least visited (most novel) states first
    state_counts.sort(key=lambda x: x[1])

    # Return the top N most novel states
    return [{'state': state, 'count': count, 'novelty_score': 1.0 / (count ** 0.5), 'topic': topic}
            for state, count in state_counts[:limit]]

def get_least_novel_states(limit=10, topic=None):
    """Get the least novel states based on visit counts."""
    if topic:
        # Get topic-specific novelty counts
        topic_counts = read_topic_novelty_counts()
        if topic not in topic_counts:
            return []
        novelty_counts = topic_counts[topic]
    else:
        # Get global novelty counts
        novelty_counts = read_novelty_counts()

    # Convert to list of (state, count) tuples
    state_counts = [(state, count) for state, count in novelty_counts.items()]

    # Sort by count (descending) to get the most visited (least novel) states first
    state_counts.sort(key=lambda x: x[1], reverse=True)

    # Return the top N least novel states
    return [{'state': state, 'count': count, 'novelty_score': 1.0 / (count ** 0.5), 'topic': topic}
            for state, count in state_counts[:limit]]

def get_novelty_statistics(topic=None):
    """Get statistics about the novelty tracking."""
    if topic:
        # Get topic-specific novelty counts
        topic_counts = read_topic_novelty_counts()
        if topic not in topic_counts:
            novelty_counts = {}
        else:
            novelty_counts = topic_counts[topic]
    else:
        # Get global novelty counts
        novelty_counts = read_novelty_counts()

    if not novelty_counts:
        return {
            "topic": topic,
            "total_states": 0,
            "total_visits": 0,
            "avg_visits_per_state": 0,
            "max_visits": 0,
            "min_visits": 0,
            "unique_states": 0,
            "avg_novelty": 0.0,
            "exploration_rate": 0.5  # Default balanced exploration rate
        }

    total_states = len(novelty_counts)
    total_visits = sum(novelty_counts.values())
    avg_visits = total_visits / total_states if total_states > 0 else 0
    max_visits = max(novelty_counts.values()) if novelty_counts else 0
    min_visits = min(novelty_counts.values()) if novelty_counts else 0
    unique_states = sum(1 for count in novelty_counts.values() if count == 1)

    # Calculate average novelty score
    import math
    avg_novelty = sum(1.0 / math.sqrt(count + 1) for count in novelty_counts.values()) / total_states if total_states > 0 else 0.0

    # Calculate exploration rate based on average novelty
    if avg_novelty > 0.7:  # Mostly novel states
        exploration_rate = 0.3  # Reduce exploration
    elif avg_novelty < 0.3:  # Mostly familiar states
        exploration_rate = 0.7  # Increase exploration
    else:
        exploration_rate = 0.5  # Balanced approach

    return {
        "topic": topic,
        "total_states": total_states,
        "total_visits": total_visits,
        "avg_visits_per_state": avg_visits,
        "max_visits": max_visits,
        "min_visits": min_visits,
        "unique_states": unique_states,
        "avg_novelty": avg_novelty,
        "exploration_rate": exploration_rate
    }

@novelty_bp.route('/api/novelty/logs', methods=['GET'])
def get_novelty_logs():
    """Get the most recent curiosity log entries."""
    max_lines = request.args.get('max_lines', 100, type=int)
    logs = read_curiosity_log(max_lines)
    return jsonify(logs)

@novelty_bp.route('/api/novelty/timeline', methods=['GET'])
def get_novelty_timeline():
    """Get a timeline of novelty scores."""
    timeline = parse_novelty_timeline()
    return jsonify(timeline)

@novelty_bp.route('/api/novelty/most-novel', methods=['GET'])
def get_most_novel():
    """Get the most novel states."""
    limit = request.args.get('limit', 10, type=int)
    topic = request.args.get('topic', None)
    states = get_most_novel_states(limit, topic)
    return jsonify(states)

@novelty_bp.route('/api/novelty/least-novel', methods=['GET'])
def get_least_novel():
    """Get the least novel states."""
    limit = request.args.get('limit', 10, type=int)
    topic = request.args.get('topic', None)
    states = get_least_novel_states(limit, topic)
    return jsonify(states)

@novelty_bp.route('/api/novelty/statistics', methods=['GET'])
def get_statistics():
    """Get statistics about the novelty tracking."""
    topic = request.args.get('topic', None)
    stats = get_novelty_statistics(topic)
    return jsonify(stats)

@novelty_bp.route('/api/novelty/topics', methods=['GET'])
def get_topics():
    """Get a list of all topics with novelty data."""
    topic_counts = read_topic_novelty_counts()
    topics = []

    for topic in topic_counts.keys():
        # Get statistics for this topic
        stats = get_novelty_statistics(topic)
        topics.append({
            "topic": topic,
            "total_states": stats["total_states"],
            "avg_novelty": stats["avg_novelty"],
            "exploration_rate": stats["exploration_rate"]
        })

    # Sort by exploration rate (descending) to prioritize topics that need more exploration
    topics.sort(key=lambda x: x["exploration_rate"], reverse=True)

    return jsonify(topics)

@novelty_bp.route('/api/novelty/data', methods=['GET'])
def get_novelty_data():
    """Get all novelty data for the dashboard."""
    max_lines = request.args.get('max_lines', 20, type=int)
    limit = request.args.get('limit', 10, type=int)
    topic = request.args.get('topic', None)

    # Get topic list
    topic_counts = read_topic_novelty_counts()
    topics = []

    for topic_name in topic_counts.keys():
        # Get statistics for this topic
        stats = get_novelty_statistics(topic_name)
        topics.append({
            "topic": topic_name,
            "total_states": stats["total_states"],
            "avg_novelty": stats["avg_novelty"],
            "exploration_rate": stats["exploration_rate"]
        })

    # Sort by exploration rate (descending) to prioritize topics that need more exploration
    topics.sort(key=lambda x: x["exploration_rate"], reverse=True)

    return jsonify({
        'timeline': parse_novelty_timeline(),
        'most_novel': get_most_novel_states(limit, topic),
        'least_novel': get_least_novel_states(limit, topic),
        'statistics': get_novelty_statistics(topic),
        'topics': topics,
        'current_topic': topic,
        'recent_logs': read_curiosity_log(max_lines)
    })
