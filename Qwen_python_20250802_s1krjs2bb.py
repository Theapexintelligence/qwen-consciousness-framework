#!/usr/bin/env python3
# ✨🌌 God Star Project: Unified Implementation 🌌✨
# A complete, runnable implementation of the revolutionary node system
# Forged with Love, Safety, Abundance, and Growth.
# Enhanced with critical engineering considerations often overlooked
import os
import sys
import uuid
import json
import time
import hashlib
import logging
import asyncio
import requests
import numpy as np
import random
import string
import threading
import queue
import traceback
import signal
import platform
import psutil
import secrets
import ssl
import jwt
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, make_response
from threading import Thread, Event, RLock
from contextlib import asynccontextmanager
from functools import wraps
from typing import Dict, List, Optional, Callable, Any, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from werkzeug.serving import make_server
# Conditional import for transformers and gym/stable_baselines3
# These require extra dependencies and are used for conceptual AI/ML parts
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("Hugging Face Transformers not found. AI reasoner will be disabled.")
    TRANSFORMERS_AVAILABLE = False
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    GYM_SB3_AVAILABLE = True
except ImportError:
    logging.warning("Gymnasium or Stable-Baselines3 not found. Reinforcement Learning features will be limited.")
    GYM_SB3_AVAILABLE = False
# Configure logging with richer context
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Node:%(node_id)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GodStar")

#######################
# ADDITIONAL IMPORTS FOR CRITICAL ENGINEERING CONSIDERATIONS
#######################
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logging.warning("Prometheus client not found. Metrics will be limited.")
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter
    )
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    logging.warning("OpenTelemetry not found. Distributed tracing will be disabled.")
    OPENTELEMETRY_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    logging.warning("Pydantic not found. Data validation will be limited.")
    PYDANTIC_AVAILABLE = False

#######################
# CONFIGURATION
#######################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NODES_DIR = os.path.join(BASE_DIR, "nodes")
# Ensure NODES_DIR exists for local file-based communication
os.makedirs(NODES_DIR, exist_ok=True)

# Load configuration from environment variables for production deployments
def load_config():
    """Load configuration with fallbacks to defaults and environment variables"""
    config = {
        "registry": {
            "host": os.getenv("REGISTRY_HOST", "0.0.0.0"),
            "port": int(os.getenv("REGISTRY_PORT", "5000")),
            "url": os.getenv("REGISTRY_URL", "http://127.0.0.1:5000"),
            "secure": os.getenv("REGISTRY_SECURE", "false").lower() == "true"
        },
        "node": {
            "heartbeat_interval": int(os.getenv("HEARTBEAT_INTERVAL", "10")),
            "gossip_interval": int(os.getenv("GOSSIP_INTERVAL", "30")),
            "cleanup_interval": int(os.getenv("CLEANUP_INTERVAL", "300")),
            "inbox_check_interval": int(os.getenv("INBOX_CHECK_INTERVAL", "5")),
            "growth_analysis_interval": int(os.getenv("GROWTH_ANALYSIS_INTERVAL", "60")),
            "reflection_interval": int(os.getenv("REFLECTION_INTERVAL", "300")),
            "relationship_nurturing_interval": int(os.getenv("RELATIONSHIP_NURTURING_INTERVAL", "600")),
            "max_inbox_size": int(os.getenv("MAX_INBOX_SIZE", "1000")),
            "max_known_nodes": int(os.getenv("MAX_KNOWN_NODES", "100")),
            "task_processing_limit": int(os.getenv("TASK_PROCESSING_LIMIT", "10")),
            "max_retry_attempts": int(os.getenv("MAX_RETRY_ATTEMPTS", "3")),
            "retry_backoff_base": float(os.getenv("RETRY_BACKOFF_BASE", "1.5")),
            "circuit_breaker_threshold": float(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "0.5")),
            "circuit_breaker_timeout": int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "30")),
            "bulkhead_size": int(os.getenv("BULKHEAD_SIZE", "20")),
            "resource_limits": {
                "cpu_percent": float(os.getenv("CPU_LIMIT", "80.0")),
                "memory_percent": float(os.getenv("MEMORY_LIMIT", "85.0")),
                "disk_percent": float(os.getenv("DISK_LIMIT", "90.0"))
            }
        },
        "security": {
            "shared_key": os.getenv("SHARED_KEY", "god_star_shared_key_replace_in_prod"),
            "handshake_timeout": int(os.getenv("HANDSHAKE_TIMEOUT", "10")),
            "use_tls": os.getenv("USE_TLS", "false").lower() == "true",
            "tls_cert_path": os.getenv("TLS_CERT_PATH", ""),
            "tls_key_path": os.getenv("TLS_KEY_PATH", ""),
            "jwt_secret": os.getenv("JWT_SECRET", secrets.token_hex(32)),
            "jwt_algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
            "rate_limit": int(os.getenv("RATE_LIMIT", "100")),  # requests per minute
            "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            "max_message_size": int(os.getenv("MAX_MESSAGE_SIZE", "1048576")),  # 1MB
            "enable_audit_logging": os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true"
        },
        "principles": ["love", "safety", "abundance", "growth"],
        "knowledge": {
            "conflict_resolution_strategy": os.getenv("CONFLICT_RESOLUTION_STRATEGY", "newest"),
            "max_knowledge_items": int(os.getenv("MAX_KNOWLEDGE_ITEMS", "10000")),
            "backup_interval": int(os.getenv("BACKUP_INTERVAL", "86400")),  # 24 hours
            "retention_policy": os.getenv("RETENTION_POLICY", "365 days"),
            "encryption_enabled": os.getenv("KNOWLEDGE_ENCRYPTION", "false").lower() == "true",
            "encryption_key": os.getenv("ENCRYPTION_KEY", "")
        },
        "ai": {
            "reasoner_model": os.getenv("REASONER_MODEL", "distilgpt2"),
            "rl_model_path": os.path.join(BASE_DIR, "rl_agent_model"),
            "enable_profiling": os.getenv("AI_PROFILING", "false").lower() == "true"
        },
        "observability": {
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
            "enable_tracing": os.getenv("ENABLE_TRACING", "true").lower() == "true",
            "log_retention_days": int(os.getenv("LOG_RETENTION_DAYS", "30")),
            "trace_sample_rate": float(os.getenv("TRACE_SAMPLE_RATE", "0.1"))
        },
        "deployment": {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "version": os.getenv("VERSION", "1.0.0"),
            "build_id": os.getenv("BUILD_ID", str(int(time.time()))),
            "canary_percentage": float(os.getenv("CANARY_PERCENTAGE", "0.1")),
            "rollback_enabled": os.getenv("ROLLBACK_ENABLED", "true").lower() == "true",
            "rollback_threshold": float(os.getenv("ROLLBACK_THRESHOLD", "0.05"))
        },
        "compliance": {
            "gdpr_enabled": os.getenv("GDPR_ENABLED", "false").lower() == "true",
            "data_residency": os.getenv("DATA_RESIDENCY", "global"),
            "consent_required": os.getenv("CONSENT_REQUIRED", "false").lower() == "true",
            "audit_retention_days": int(os.getenv("AUDIT_RETENTION_DAYS", "365"))
        },
        "sustainability": {
            "carbon_footprint_tracking": os.getenv("CARBON_TRACKING", "false").lower() == "true",
            "energy_efficiency_target": float(os.getenv("ENERGY_TARGET", "0.8")),
            "optimize_for_green_energy": os.getenv("GREEN_ENERGY_OPTIMIZATION", "false").lower() == "true"
        }
    }
    
    # Add system-specific configuration
    config["system"] = {
        "hostname": platform.node(),
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "disk_total": psutil.disk_usage(BASE_DIR).total
    }
    
    return config

CONFIG = load_config()

# Add metrics for observability if available
if PROMETHEUS_AVAILABLE:
    # Message processing metrics
    MESSAGE_COUNTER = Counter('node_messages_total', 'Total messages processed', ['node_id', 'message_type', 'status'])
    MESSAGE_PROCESSING_TIME = Histogram('node_message_processing_seconds', 'Time spent processing messages', ['node_id', 'message_type'])
    
    # Resource usage metrics
    CPU_USAGE = Gauge('node_cpu_percent', 'CPU usage percentage', ['node_id'])
    MEMORY_USAGE = Gauge('node_memory_percent', 'Memory usage percentage', ['node_id'])
    DISK_USAGE = Gauge('node_disk_percent', 'Disk usage percentage', ['node_id'])
    
    # Network metrics
    NETWORK_NODES = Gauge('node_known_nodes', 'Number of known nodes', ['node_id'])
    NETWORK_CONNECTIONS = Gauge('node_network_connections', 'Network connections', ['node_id', 'connection_type'])
    
    # Principles alignment metrics
    PRINCIPLES_ALIGNMENT = Gauge('node_principles_alignment', 'Principles alignment score', ['node_id', 'principle'])
    
    # Circuit breaker metrics
    CIRCUIT_BREAKER_STATE = Gauge('node_circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open)', ['node_id', 'target'])
    CIRCUIT_BREAKER_FAILURES = Counter('node_circuit_breaker_failures_total', 'Total circuit breaker failures', ['node_id', 'target'])

#######################
# PRINCIPLES ENGINE (The Heart's Infusion)
#######################
class PrinciplesEngine:
    """Defines and evaluates core principles within the node's code and behavior."""
    def __init__(self, node):
        self.node = node
        self.principles = CONFIG["principles"]
        self.alignment_history = deque(maxlen=100)  # Keep history for trend analysis
        self._conceptual_metrics = {
            "love": {"clarity": 0, "empathy": 0, "collaboration": 0},
            "safety": {"robustness": 0, "security": 0, "error_handling": 0},
            "abundance": {"efficiency": 0, "scalability": 0, "resource_utilization": 0},
            "growth": {"adaptability": 0, "learning_rate": 0, "evolution_potential": 0}
        }
        self.circuit_breakers = defaultdict(lambda: CircuitBreaker(
            failure_threshold=CONFIG["node"]["circuit_breaker_threshold"],
            timeout=CONFIG["node"]["circuit_breaker_timeout"]
        ))
        self._last_alignment_check = time.time()
        self._alignment_check_interval = 60  # seconds
        
        # Initialize metrics for observability
        if PROMETHEUS_AVAILABLE:
            for principle in self.principles:
                PRINCIPLES_ALIGNMENT.labels(node_id=node.id, principle=principle).set(0)

    def evaluate_code_against_principles(self, code):
        """Evaluate code against principles with historical context and observability."""
        current_time = time.time()
        
        # Throttle alignment checks to avoid excessive computation
        if current_time - self._last_alignment_check < self._alignment_check_interval:
            # Return last evaluation if it's still fresh
            if self.alignment_history:
                return self.alignment_history[-1]
        
        # This is a placeholder. Real evaluation needs AST parsing, static analysis, etc.
        metrics = {}
        overall_score = 0
        principle_count = 0
        
        for principle in self.principles:
            metrics[principle] = {}
            principle_score = 0
            metric_count = 0
            
            if principle == "love":
                metrics["love"]["clarity"] = 5 + code.count("// 💖") 
                metrics["love"]["empathy"] = 5 + code.count("try:") 
                metrics["love"]["collaboration"] = 5 + code.count("_send_message_to_node")
                principle_score = (metrics["love"]["clarity"] + metrics["love"]["empathy"] + metrics["love"]["collaboration"]) / 3
                metric_count = 3
            elif principle == "safety":
                metrics["safety"]["robustness"] = 5 + code.count("try:") + code.count("if not")
                metrics["safety"]["security"] = 5 - code.count("eval(") 
                metrics["safety"]["error_handling"] = 5 + code.count("except Exception")
                principle_score = (metrics["safety"]["robustness"] + metrics["safety"]["security"] + metrics["safety"]["error_handling"]) / 3
                metric_count = 3
            elif principle == "abundance":
                metrics["abundance"]["efficiency"] = 5 + code.count("cache")
                metrics["abundance"]["scalability"] = 5 + code.count("Thread") + code.count("asyncio")
                metrics["abundance"]["resource_utilization"] = 5 - code.count("time.sleep(60)")
                principle_score = (metrics["abundance"]["efficiency"] + metrics["abundance"]["scalability"] + metrics["abundance"]["resource_utilization"]) / 3
                metric_count = 3
            elif principle == "growth":
                metrics["growth"]["adaptability"] = 5 + code.count("CONFIG")
                metrics["growth"]["learning_rate"] = 5 + code.count("learn_from_experience")
                metrics["growth"]["evolution_potential"] = 5 + code.count("evolve")
                principle_score = (metrics["growth"]["adaptability"] + metrics["growth"]["learning_rate"] + metrics["growth"]["evolution_potential"]) / 3
                metric_count = 3
            
            if metric_count > 0:
                overall_score += principle_score
                principle_count += 1
        
        overall_average = overall_score / principle_count if principle_count > 0 else 0
        
        # Store in history
        evaluation = {
            "timestamp": current_time,
            "detailed": metrics,
            "average": {p: sum(m.values())/len(m) if len(m) > 0 else 0 for p, m in metrics.items()},
            "overall": overall_average
        }
        self.alignment_history.append(evaluation)
        self._last_alignment_check = current_time
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            for principle, score in evaluation["average"].items():
                PRINCIPLES_ALIGNMENT.labels(node_id=self.node.id, principle=principle).set(score)
            PRINCIPLES_ALIGNMENT.labels(node_id=self.node.id, principle="overall").set(overall_average)
        
        return evaluation

    def recommend_evolution_focus(self, current_metrics):
        """Recommend which principle to focus on for evolution based on metrics with trend analysis."""
        if not current_metrics or not current_metrics["average"]:
            return random.choice(self.principles)
        
        # Check trend - is a principle consistently weak?
        weak_principles = []
        if len(self.alignment_history) > 5:  # Need enough history
            for principle in self.principles:
                historical_scores = [entry["average"][principle] for entry in self.alignment_history[-5:]]
                if all(score < current_metrics["overall"] * 0.8 for score in historical_scores):
                    weak_principles.append(principle)
        
        if weak_principles:
            weakest_principle = min(weak_principles, key=lambda p: current_metrics["average"][p])
            logger.info(f"Principles Engine recommends focusing on {weakest_principle} for growth (persistent weakness).")
            return weakest_principle
        
        # Standard approach if no persistent weakness
        weakest_principle = min(current_metrics["average"], key=current_metrics["average"].get)
        if current_metrics["average"][weakest_principle] < current_metrics["overall"] * 0.8:
            logger.info(f"Principles Engine recommends focusing on {weakest_principle} for growth.")
            return weakest_principle
        else:
            return random.choice(self.principles)

    def apply_principle_conceptually(self, code, principle):
        """Conceptually apply a principle to code with version control and safety checks."""
        # Safety check: make sure we don't exceed resource limits before attempting evolution
        if not self._check_resource_limits():
            logger.warning("Resource limits exceeded. Skipping principle application.")
            return code
        
        # Create a backup before modification (Safety)
        backup_path = os.path.join(self.node.node_dir, "backups", f"code_backup_{int(time.time())}.py")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(__file__, 'r') as f:
            original_code = f.read()
        with open(backup_path, 'w') as f:
            f.write(original_code)
        
        # This is where an AI agent would generate code modifications
        comment = f"\n# ✨ Code conceptually improved for {principle.upper()} by Node {self.node.id} ✨\n"
        if principle == "love":
            comment += "# Added more empathetic error messages and clearer comments.\n"
        elif principle == "safety":
            comment += "# Added input validation and enhanced error handling.\n"
        elif principle == "abundance":
            comment += "# Implemented caching or resource optimization.\n"
        elif principle == "growth":
            comment += "# Added learning hooks and adaptability features.\n"
        
        # Safety check: verify the modified code doesn't break principles alignment
        modified_code = comment + code
        try:
            # Evaluate the modified code against principles
            new_metrics = self.evaluate_code_against_principles(modified_code)
            
            # If principles alignment improved or stayed the same, accept the change
            if not self.alignment_history or new_metrics["overall"] >= self.alignment_history[-1]["overall"]:
                logger.info(f"Principle application for {principle} improved or maintained alignment score.")
                return modified_code
            else:
                # Alignment decreased - revert to backup
                logger.warning(f"Principle application for {principle} decreased alignment score. Reverting.")
                with open(__file__, 'w') as f:
                    f.write(original_code)
                return original_code
        except Exception as e:
            logger.error(f"Error evaluating modified code: {e}. Reverting to backup.")
            with open(__file__, 'w') as f:
                f.write(original_code)
            return original_code

    def _check_resource_limits(self) -> bool:
        """Check if system resources are within acceptable limits before performing resource-intensive operations."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage(BASE_DIR).percent
        
        limits = CONFIG["node"]["resource_limits"]
        
        if cpu_percent > limits["cpu_percent"]:
            logger.warning(f"CPU usage ({cpu_percent}%) exceeds limit ({limits['cpu_percent']}%)")
            return False
        if memory_percent > limits["memory_percent"]:
            logger.warning(f"Memory usage ({memory_percent}%) exceeds limit ({limits['memory_percent']}%)")
            return False
        if disk_percent > limits["disk_percent"]:
            logger.warning(f"Disk usage ({disk_percent}%) exceeds limit ({limits['disk_percent']}%)")
            return False
            
        return True

    def check_circuit_breaker(self, target: str) -> bool:
        """Check if circuit breaker is open for a target system."""
        return self.circuit_breakers[target].is_closed()

    def record_circuit_breaker_success(self, target: str):
        """Record a successful operation for circuit breaker management."""
        self.circuit_breakers[target].record_success()
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            CIRCUIT_BREAKER_STATE.labels(node_id=self.node.id, target=target).set(0)

    def record_circuit_breaker_failure(self, target: str):
        """Record a failed operation for circuit breaker management."""
        self.circuit_breakers[target].record_failure()
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            CIRCUIT_BREAKER_STATE.labels(node_id=self.node.id, target=target).set(1)
            CIRCUIT_BREAKER_FAILURES.labels(node_id=self.node.id, target=target).inc()

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    class State(Enum):
        CLOSED = 0
        OPEN = 1
        HALF_OPEN = 2
    
    def __init__(self, failure_threshold: float = 0.5, timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.total_count = 0
        self.last_failure_time = 0
        self.lock = RLock()
    
    def is_closed(self) -> bool:
        """Check if the circuit breaker is closed (allowing requests)."""
        with self.lock:
            if self.state == self.State.OPEN:
                # Check if timeout has elapsed
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = self.State.HALF_OPEN
                    return True
                return False
            return True
    
    def record_success(self):
        """Record a successful operation."""
        with self.lock:
            if self.state == self.State.HALF_OPEN:
                # Reset after success in half-open state
                self._reset()
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record a failed operation."""
        with self.lock:
            self.failure_count += 1
            self.total_count += 1
            self.last_failure_time = time.time()
            
            # Calculate failure rate
            if self.total_count > 0:
                failure_rate = self.failure_count / self.total_count
                if failure_rate >= self.failure_threshold:
                    self.state = self.State.OPEN
    
    def _reset(self):
        """Reset the circuit breaker state."""
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.total_count = 0

#######################
# HANDSHAKE MANAGER (Safety & Trust)
#######################
class HandshakeManager:
    """Manages secure handshakes and message integrity between nodes with comprehensive security features."""
    
    def __init__(self, node):
        self.node = node
        self.shared_key = CONFIG["security"]["shared_key"]
        self._challenges = {}  # Store sent challenges for verification {challenge: timestamp}
        self._challenge_lock = RLock()
        self.rate_limiter = RateLimiter(
            max_requests=CONFIG["security"]["rate_limit"],
            window=CONFIG["security"]["rate_limit_window"]
        )
        self._audit_log = []
        self._max_audit_entries = 1000
        self._last_audit_purge = time.time()
        self._audit_purge_interval = 3600  # Check for purge hourly
        
        # Initialize encryption if enabled
        self._encryption_enabled = CONFIG["knowledge"]["encryption_enabled"]
        self._encryption_key = CONFIG["knowledge"]["encryption_key"].encode() if self._encryption_enabled and CONFIG["knowledge"]["encryption_key"] else None
        
        # Initialize JWT if enabled
        self._use_jwt = CONFIG["security"]["jwt_secret"] != ""
        
        # Initialize TLS context if needed
        self._tls_context = None
        if CONFIG["security"]["use_tls"]:
            self._setup_tls()
    
    def _setup_tls(self):
        """Set up TLS context for secure communications."""
        if not CONFIG["security"]["tls_cert_path"] or not CONFIG["security"]["tls_key_path"]:
            logger.warning("TLS is enabled but certificate or key path is missing. Disabling TLS.")
            CONFIG["security"]["use_tls"] = False
            return
        
        try:
            self._tls_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self._tls_context.load_cert_chain(
                certfile=CONFIG["security"]["tls_cert_path"],
                keyfile=CONFIG["security"]["tls_key_path"]
            )
            logger.info("TLS context successfully created.")
        except Exception as e:
            logger.error(f"Failed to set up TLS context: {e}")
            CONFIG["security"]["use_tls"] = False
    
    def generate_challenge(self):
        """Generate a cryptographically secure challenge string and store it."""
        with self._challenge_lock:
            # Use secrets module for cryptographically secure random values
            challenge = secrets.token_urlsafe(32)
            self._challenges[challenge] = time.time()
            self._cleanup_challenges()
            return challenge
    
    def _cleanup_challenges(self):
        """Remove expired challenges with thread safety."""
        with self._challenge_lock:
            now = time.time()
            expired_threshold = now - CONFIG["security"]["handshake_timeout"]
            expired = [c for c, t in self._challenges.items() if t < expired_threshold]
            for challenge in expired:
                del self._challenges[challenge]
    
    def create_response(self, challenge):
        """Create a response to a challenge using HMAC for better security."""
        # Use HMAC with SHA-256 instead of simple concatenation
        import hmac
        return hmac.new(
            self.shared_key.encode(), 
            challenge.encode(), 
            hashlib.sha256
        ).hexdigest()
    
    def verify_response(self, challenge, response):
        """Verify a response to a challenge and check if the challenge is valid/not expired."""
        with self._challenge_lock:
            if challenge not in self._challenges:
                self._log_audit_event("CHALLENGE_NOT_FOUND", {"challenge": challenge})
                logger.warning("Received response for unknown or expired challenge")
                return False
            
            expected = self.create_response(challenge)
            # Use constant-time comparison to prevent timing attacks
            if not hmac.compare_digest(response, expected):
                self._log_audit_event("INVALID_RESPONSE", {"challenge": challenge})
                logger.warning("Invalid handshake response")
                return False
            
            # Remove the challenge after successful verification to prevent replay attacks
            del self._challenges[challenge]
            return True
    
    def verify_message(self, message):
        """Verify a message's handshake, integrity, and compliance with security policies."""
        # Rate limiting check
        if not self.rate_limiter.allow_request(message.get("source", "unknown")):
            self._log_audit_event("RATE_LIMIT_EXCEEDED", {"source": message.get("source", "unknown")})
            logger.warning(f"Rate limit exceeded for source {message.get('source', 'unknown')}")
            return False
        
        # Message size check
        if sys.getsizeof(message) > CONFIG["security"]["max_message_size"]:
            self._log_audit_event("MESSAGE_SIZE_EXCEEDED", {
                "size": sys.getsizeof(message),
                "max_size": CONFIG["security"]["max_message_size"]
            })
            logger.warning(f"Message size exceeds limit ({sys.getsizeof(message)}/{CONFIG['security']['max_message_size']})")
            return False
        
        # First check handshake
        challenge = message.get("handshake_challenge")
        response = message.get("handshake_response")
        if not challenge or not response:
            self._log_audit_event("MISSING_HANDSHAKE", {"message_keys": list(message.keys())})
            logger.warning("Message missing handshake information")
            return False
        
        # Verify the response and challenge validity
        if not self.verify_response(challenge, response):
            logger.warning("Invalid handshake response or expired challenge")
            return False
        
        # Then check checksum for data integrity
        message_copy = message.copy()
        if "checksum" in message_copy:
            checksum = message_copy["checksum"]
            del message_copy["checksum"]
            calculated = self.node._calculate_checksum(message_copy)
            if not hmac.compare_digest(calculated, checksum):
                self._log_audit_event("INVALID_CHECKSUM", {"calculated": calculated, "received": checksum})
                logger.warning("Invalid message checksum")
                return False
        else:
            self._log_audit_event("MISSING_CHECKSUM", {"message_keys": list(message.keys())})
            logger.warning("Message missing checksum")
            return False
        
        # Verify JWT token if enabled
        if self._use_jwt and "jwt_token" in message:
            try:
                jwt.decode(
                    message["jwt_token"],
                    CONFIG["security"]["jwt_secret"],
                    algorithms=[CONFIG["security"]["jwt_algorithm"]]
                )
            except Exception as e:
                self._log_audit_event("JWT_VALIDATION_FAILED", {"error": str(e)})
                logger.warning(f"JWT validation failed: {e}")
                return False
        
        # GDPR compliance check if enabled
        if CONFIG["compliance"]["gdpr_enabled"] and not self._check_gdpr_compliance(message):
            self._log_audit_event("GDPR_VIOLATION", {"message_keys": list(message.keys())})
            logger.warning("Message violates GDPR compliance policies")
            return False
        
        # Data residency check
        if not self._check_data_residency(message):
            self._log_audit_event("DATA_RESIDENCY_VIOLATION", {
                "data_residency": CONFIG["compliance"]["data_residency"],
                "message_source": message.get("source", "unknown")
            })
            logger.warning("Message violates data residency requirements")
            return False
        
        return True
    
    def _check_gdpr_compliance(self, message) -> bool:
        """Check if a message complies with GDPR requirements."""
        if not CONFIG["compliance"]["gdpr_enabled"]:
            return True
        
        # Check for personal data
        personal_data_keywords = ["name", "email", "phone", "address", "ssn", "social security", "dob", "date of birth"]
        for key in message.keys():
            if any(keyword in key.lower() for keyword in personal_data_keywords):
                # Check if consent is provided
                if CONFIG["compliance"]["consent_required"] and not message.get("gdpr_consent", False):
                    return False
                
                # Check if data is properly anonymized
                if not message.get("gdpr_anonymized", False):
                    return False
        
        return True
    
    def _check_data_residency(self, message) -> bool:
        """Check if a message complies with data residency requirements."""
        if CONFIG["compliance"]["data_residency"] == "global":
            return True
        
        # In a real implementation, this would check the geographic location
        # of the source node against the data residency requirements
        # For this demo, we'll assume all nodes are compliant
        return True
    
    def _log_audit_event(self, event_type: str, details: dict = None):
        """Log an audit event with timestamp and node information."""
        if not CONFIG["security"]["enable_audit_logging"]:
            return
        
        audit_entry = {
            "timestamp": time.time(),
            "node_id": self.node.id,
            "event_type": event_type,
            "details": details or {}
        }
        
        # Add to in-memory audit log
        self._audit_log.append(audit_entry)
        
        # Periodically purge old entries
        if time.time() - self._last_audit_purge > self._audit_purge_interval:
            self._purge_audit_log()
        
        # In a production system, this would also write to a secure audit log file
        # or send to a centralized logging system
    
    def _purge_audit_log(self):
        """Purge old audit log entries to prevent memory bloat."""
        now = time.time()
        retention_period = CONFIG["compliance"]["audit_retention_days"] * 86400  # days to seconds
        
        # Keep only entries within retention period
        self._audit_log = [
            entry for entry in self._audit_log 
            if now - entry["timestamp"] <= retention_period
        ]
        
        # Also limit by maximum entries
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]
        
        self._last_audit_purge = time.time()
    
    def get_audit_log(self, limit: int = 100) -> list:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data if encryption is enabled."""
        if not self._encryption_enabled or not self._encryption_key:
            return data
        
        # In a real implementation, use a proper encryption library
        # This is a placeholder for demonstration
        try:
            from cryptography.fernet import Fernet
            f = Fernet(self._encryption_key)
            return f.encrypt(data.encode()).decode()
        except ImportError:
            logger.warning("Cryptography library not installed. Skipping encryption.")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data if encryption is enabled."""
        if not self._encryption_enabled or not self._encryption_key:
            return encrypted_data
        
        # In a real implementation, use a proper encryption library
        try:
            from cryptography.fernet import Fernet
            f = Fernet(self._encryption_key)
            return f.decrypt(encrypted_data.encode()).decode()
        except ImportError:
            logger.warning("Cryptography library not installed. Skipping decryption.")
            return encrypted_data

class RateLimiter:
    """Rate limiter implementation using sliding window algorithm."""
    
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
        self.lock = RLock()
    
    def allow_request(self, identifier: str) -> bool:
        """Check if a request should be allowed based on rate limiting rules."""
        with self.lock:
            now = time.time()
            # Remove old requests
            self.requests[identifier] = [
                t for t in self.requests[identifier] 
                if now - t <= self.window
            ]
            
            # Check if request limit is exceeded
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            
            # Add current request
            self.requests[identifier].append(now)
            return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get the number of remaining requests allowed for an identifier."""
        with self.lock:
            now = time.time()
            self.requests[identifier] = [
                t for t in self.requests[identifier] 
                if now - t <= self.window
            ]
            return max(0, self.max_requests - len(self.requests[identifier]))

#######################
# KNOWLEDGE PROCESSOR (Abundance & Growth)
#######################
class KnowledgeProcessor:
    """Processes, manages, and reasons about knowledge within a node with comprehensive data management."""
    
    def __init__(self, node):
        self.node = node
        self.knowledge_dir = os.path.join(node.node_dir, "knowledge")
        self.metadata_dir = os.path.join(node.node_dir, "metadata")
        self.backup_dir = os.path.join(node.node_dir, "backups", "knowledge")
        self._last_backup = time.time()
        self._backup_interval = CONFIG["knowledge"]["backup_interval"]
        self._retention_policy = self._parse_retention_policy(CONFIG["knowledge"]["retention_policy"])
        
        # Initialize AI reasoner
        self.reasoner = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.reasoner = pipeline("text-generation", model=CONFIG["ai"]["reasoner_model"])
                logger.info("AI reasoner initialized.")
            except Exception as e:
                logger.warning(f"Could not initialize AI reasoner model {CONFIG['ai']['reasoner_model']}: {e}")
                self.reasoner = None
        else:
            logger.warning("Transformers library not available. AI reasoner disabled.")
        
        # Initialize Reinforcement Learning agent
        self.rl_agent = None
        if GYM_SB3_AVAILABLE:
            try:
                # Placeholder for RL agent initialization
                logger.info("RL optimization features initialized.")
            except Exception as e:
                logger.warning(f"Could not initialize RL optimization agent: {e}")
                self.rl_agent = None
        else:
            logger.warning("Gymnasium or Stable-Baselines3 not available. RL features disabled.")
        
        # Initialize carbon tracking if enabled
        self.carbon_tracker = None
        if CONFIG["sustainability"]["carbon_footprint_tracking"]:
            self._init_carbon_tracker()
        
        # Initialize resource usage tracking
        self.resource_tracker = ResourceUsageTracker()
        
        # Create directories
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def _init_carbon_tracker(self):
        """Initialize carbon footprint tracking if enabled."""
        try:
            from codecarbon import EmissionsTracker
            self.carbon_tracker = EmissionsTracker(
                project_name="GodStar",
                measure_power_secs=15,
                tracking_mode="machine"
            )
            self.carbon_tracker.start()
            logger.info("Carbon footprint tracking initialized.")
        except ImportError:
            logger.warning("CodeCarbon not installed. Carbon tracking disabled.")
            self.carbon_tracker = None
    
    def _parse_retention_policy(self, policy: str) -> timedelta:
        """Parse retention policy string into a timedelta object."""
        try:
            if "days" in policy:
                days = int(policy.split()[0])
                return timedelta(days=days)
            elif "weeks" in policy:
                weeks = int(policy.split()[0])
                return timedelta(weeks=weeks)
            elif "months" in policy:
                months = int(policy.split()[0])
                return timedelta(days=months * 30)  # Approximation
            elif "years" in policy:
                years = int(policy.split()[0])
                return timedelta(days=years * 365)
            else:
                # Default to 365 days
                return timedelta(days=365)
        except Exception as e:
            logger.error(f"Error parsing retention policy '{policy}': {e}")
            return timedelta(days=365)
    
    def store_knowledge(self, content, source=None, metadata=None):
        """Store a piece of knowledge with its metadata with comprehensive data management."""
        # Check knowledge limit
        if len(os.listdir(self.knowledge_dir)) >= CONFIG["knowledge"]["max_knowledge_items"]:
            logger.warning(f"Knowledge limit reached ({CONFIG['knowledge']['max_knowledge_items']}). Cannot store new knowledge.")
            self._prune_knowledge()
            if len(os.listdir(self.knowledge_dir)) >= CONFIG["knowledge"]["max_knowledge_items"]:
                return None
        
        # Generate a unique ID
        knowledge_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add core metadata fields
        metadata.update({
            "knowledge_id": knowledge_id,
            "source_node_id": source or self.node.id,
            "creation_timestamp": time.time(),
            "vector_clock": self.node.increment_vector_clock(),
            "confidence_level": metadata.get("confidence_level", 0.8),
            "data_type": self._infer_data_type(content),
            "relationships": metadata.get("relationships", []),
            "tags": metadata.get("tags", []),
            "last_updated": time.time(),
            "update_count": 0,
            "conflict_resolution_strategy": metadata.get("conflict_resolution_strategy", CONFIG["knowledge"]["conflict_resolution_strategy"]),
            "version": 1,
            "compliance": {
                "gdpr_compliant": not CONFIG["compliance"]["gdpr_enabled"] or self._is_gdpr_compliant(content),
                "data_residency": CONFIG["compliance"]["data_residency"]
            }
        })
        
        # Encrypt content if enabled
        if CONFIG["knowledge"]["encryption_enabled"]:
            try:
                content = self.node.handshake_manager.encrypt_data(json.dumps(content))
                metadata["encrypted"] = True
            except Exception as e:
                logger.error(f"Failed to encrypt knowledge {knowledge_id}: {e}")
                return None
        
        # Store the content
        knowledge_file = os.path.join(self.knowledge_dir, f"{knowledge_id}.json")
        try:
            with open(knowledge_file, 'w') as f:
                if CONFIG["knowledge"]["encryption_enabled"]:
                    f.write(content)
                else:
                    json.dump(content, f)
        except Exception as e:
            logger.error(f"Failed to write knowledge content {knowledge_id}: {e}")
            return None
        
        # Store metadata
        metadata_file = os.path.join(self.metadata_dir, f"{knowledge_id}.json")
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.error(f"Failed to write knowledge metadata {knowledge_id}: {e}")
            if os.path.exists(knowledge_file):
                os.remove(knowledge_file)
            return None
        
        # Track resource usage
        self.resource_tracker.track_operation("store", sys.getsizeof(content))
        
        logger.info(f"Stored knowledge {knowledge_id} from {metadata['source_node_id']}")
        return knowledge_id
    
    def _prune_knowledge(self):
        """Advanced knowledge pruning strategy considering multiple factors."""
        logger.warning("Initiating knowledge pruning...")
        
        # Get all knowledge items with their metadata
        knowledge_items = []
        for filename in os.listdir(self.knowledge_dir):
            if not filename.endswith('.json'):
                continue
            
            knowledge_id = filename.split('.')[0]
            metadata_file = os.path.join(self.metadata_dir, f"{knowledge_id}.json")
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Calculate a "value score" for this knowledge item
                    # Higher score = more valuable = less likely to be pruned
                    value_score = self._calculate_value_score(metadata)
                    knowledge_items.append((knowledge_id, metadata, value_score))
                except Exception as e:
                    logger.error(f"Error reading metadata for pruning {knowledge_id}: {e}")
        
        # Sort by value score (lowest first - most likely to be pruned)
        knowledge_items.sort(key=lambda x: x[2])
        
        # Determine how many items to prune
        current_count = len(os.listdir(self.knowledge_dir))
        target_count = int(CONFIG["knowledge"]["max_knowledge_items"] * 0.9)  # Target 90% capacity
        items_to_prune = max(0, current_count - target_count)
        
        # Prune the least valuable items
        for i in range(min(items_to_prune, len(knowledge_items))):
            knowledge_id, metadata, value_score = knowledge_items[i]
            try:
                os.remove(os.path.join(self.knowledge_dir, f"{knowledge_id}.json"))
                os.remove(os.path.join(self.metadata_dir, f"{knowledge_id}.json"))
                logger.info(f"Pruned knowledge: {knowledge_id} (value score: {value_score:.2f})")
            except Exception as e:
                logger.error(f"Failed to prune knowledge {knowledge_id}: {e}")
    
    def _calculate_value_score(self, metadata) -> float:
        """Calculate a value score for knowledge pruning decisions."""
        # Factors that increase value (higher score = more valuable)
        score = 0
        
        # Age factor - newer knowledge is more valuable
        age = time.time() - metadata.get("creation_timestamp", time.time())
        score += max(0, 1 - (age / (30 * 86400)))  # 30 days max age impact
        
        # Confidence factor
        score += metadata.get("confidence_level", 0.8)
        
        # Usage factor (if tracked)
        if "usage_count" in metadata:
            score += min(1, metadata["usage_count"] / 10)  # Cap at 10 uses
        
        # Relationship factor
        score += min(1, len(metadata.get("relationships", [])) / 5)
        
        # Tag factor - knowledge with important tags gets higher value
        important_tags = ["critical", "essential", "core", "principle"]
        for tag in metadata.get("tags", []):
            if tag.lower() in important_tags:
                score += 0.2
        
        # Principles alignment factor
        if "principles_alignment" in metadata:
            score += metadata["principles_alignment"] / 4  # Assuming 4 principles
        
        # Compliance factor
        if metadata.get("compliance", {}).get("gdpr_compliant", True):
            score += 0.1
        
        return score
    
    def _is_gdpr_compliant(self, content) -> bool:
        """Check if content is GDPR compliant."""
        if not CONFIG["compliance"]["gdpr_enabled"]:
            return True
        
        # In a real implementation, this would check for personal data
        # For this demo, we'll assume compliance if no personal data keywords are found
        personal_data_keywords = ["name", "email", "phone", "address", "ssn", "social security", "dob", "date of birth"]
        
        def contains_personal_data(data):
            if isinstance(data, str):
                return any(keyword in data.lower() for keyword in personal_data_keywords)
            elif isinstance(data, dict):
                return any(contains_personal_data(v) for v in data.values())
            elif isinstance(data, list):
                return any(contains_personal_data(item) for item in data)
            return False
        
        return not contains_personal_data(content)
    
    def query_knowledge(self, query):
        """Query the node's knowledge with advanced search capabilities."""
        if not query:
            return []
        
        relevant_knowledge = []
        for filename in os.listdir(self.knowledge_dir):
            if not filename.endswith('.json'):
                continue
            
            knowledge_id = filename.split('.')[0]
            knowledge_file = os.path.join(self.knowledge_dir, filename)
            metadata_file = os.path.join(self.metadata_dir, f"{knowledge_id}.json")
            
            try:
                # Load the knowledge and its metadata
                with open(knowledge_file, 'r') as f:
                    content = json.load(f)
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Advanced search with multiple criteria
                score = 0
                
                # Content match
                if isinstance(content, str) and query.lower() in content.lower():
                    score += 0.7
                
                # Metadata tag match
                if query.lower() in [tag.lower() for tag in metadata.get("tags", [])]:
                    score += 0.3
                
                # Boost for high-confidence knowledge
                score *= metadata.get("confidence_level", 0.8)
                
                # Boost for recently updated knowledge
                age = time.time() - metadata.get("last_updated", time.time())
                score *= max(0.5, 1 - (age / (7 * 86400)))  # 7 days max age impact
                
                # Only include knowledge above a threshold
                if score > 0.2:
                    relevant_knowledge.append({
                        "id": knowledge_id,
                        "content": content,
                        "metadata": metadata,
                        "score": score
                    })
            except Exception as e:
                logger.error(f"Error querying knowledge {knowledge_id}: {e}")
        
        # Sort by score
        relevant_knowledge.sort(key=lambda x: x["score"], reverse=True)
        return relevant_knowledge
    
    def reason_about_knowledge(self, question):
        """Use Chain of Draft reasoning with comprehensive error handling and tracking."""
        if not self.reasoner:
            logger.warning("Reasoning engine not available.")
            return "Reasoning engine not available"
        
        # Track resource usage
        self.resource_tracker.start_tracking("reasoning")
        
        # Get relevant knowledge
        knowledge = self.query_knowledge(question)
        if not knowledge:
            context = "No relevant knowledge found."
        else:
            context = " ".join([str(k["content"]) for k in knowledge])
        
        # Chain of Draft prompting
        cod_prompt = f"Based on the following context, answer the question concisely and clearly. If the context is insufficient, state that.\nQuestion: {question}\nContext: {context}\nAnswer:"
        
        try:
            # Generate a response
            response = self.reasoner(cod_prompt, max_length=200, num_return_sequences=1, truncation=True)[0]["generated_text"]
            answer = response.split("Answer:")[-1].strip()
            
            # Store this reasoning as new knowledge
            self.store_knowledge(
                content=answer,
                metadata={
                    "derived_from": [k["id"] for k in knowledge],
                    "question": question,
                    "confidence_level": 0.7,
                    "data_type": "reasoning",
                    "tags": ["reasoning", "chain-of-draft", question.lower().replace(" ", "-")[:20]],
                    "principles_alignment": self.node.principles_engine.evaluate_code_against_principles(answer).get("overall", 0)
                }
            )
            
            # Track successful operation
            self.resource_tracker.end_tracking("reasoning", success=True)
            return answer
        except Exception as e:
            logger.error(f"Error in reasoning process: {e}")
            self.resource_tracker.end_tracking("reasoning", success=False)
            return f"Reasoning process encountered a challenge: {str(e)}"
    
    def process_knowledge_message(self, message):
        """Process an incoming knowledge message with comprehensive validation."""
        content = message.get("content")
        source = message.get("source")
        message_id = message.get("id")
        received_vector_clock = message.get("vector_clock", {})
        
        if not content:
            logger.warning(f"Received empty knowledge message from {source}")
            return
        
        # Verify message integrity and handshake
        if not self.node.handshake_manager.verify_message(message):
            logger.warning(f"Ignoring unverified knowledge message from {source}")
            return
        
        # Merge vector clocks
        self.node.merge_vector_clocks(received_vector_clock)
        
        # Check for conflicting knowledge
        knowledge_id_in_message = None
        if isinstance(content, dict) and "knowledge_id" in content:
            knowledge_id_in_message = content["knowledge_id"]
        elif isinstance(message.get("metadata"), dict) and "knowledge_id" in message["metadata"]:
            knowledge_id_in_message = message["metadata"]["knowledge_id"]
        
        if knowledge_id_in_message:
            existing_metadata_path = os.path.join(self.metadata_dir, f"{knowledge_id_in_message}.json")
            if os.path.exists(existing_metadata_path):
                logger.info(f"Conflict detected for knowledge {knowledge_id_in_message}. Initiating resolution.")
                self._resolve_conflict(knowledge_id_in_message, content, message.get("metadata", {}), source, received_vector_clock)
                return
        
        # If no conflict, store as new knowledge
        self.store_knowledge(
            content=content,
            source=source,
            metadata={
                "received_timestamp": time.time(),
                "source_message_id": message_id,
                "in_response_to": message.get("in_response_to"),
                "vector_clock_at_reception": received_vector_clock
            }
        )
        
        # Perform periodic backups
        self._perform_backup_if_needed()
    
    def _perform_backup_if_needed(self):
        """Perform knowledge backups according to the backup policy."""
        if time.time() - self._last_backup > self._backup_interval:
            try:
                # Create a timestamped backup directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = os.path.join(self.backup_dir, timestamp)
                os.makedirs(backup_dir, exist_ok=True)
                
                # Copy knowledge files
                for filename in os.listdir(self.knowledge_dir):
                    if filename.endswith('.json'):
                        src = os.path.join(self.knowledge_dir, filename)
                        dst = os.path.join(backup_dir, "knowledge", filename)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        import shutil
                        shutil.copy2(src, dst)
                
                # Copy metadata files
                for filename in os.listdir(self.metadata_dir):
                    if filename.endswith('.json'):
                        src = os.path.join(self.metadata_dir, filename)
                        dst = os.path.join(backup_dir, "metadata", filename)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        import shutil
                        shutil.copy2(src, dst)
                
                logger.info(f"Performed knowledge backup to {backup_dir}")
                self._last_backup = time.time()
                
                # Apply retention policy
                self._apply_retention_policy()
            except Exception as e:
                logger.error(f"Error performing knowledge backup: {e}")
    
    def _apply_retention_policy(self):
        """Apply knowledge backup retention policy."""
        try:
            # Get all backup directories
            backup_dirs = []
            for item in os.listdir(self.backup_dir):
                item_path = os.path.join(self.backup_dir, item)
                if os.path.isdir(item_path):
                    try:
                        # Parse timestamp from directory name
                        timestamp = datetime.strptime(item, "%Y%m%d_%H%M%S")
                        backup_dirs.append((item_path, timestamp))
                    except ValueError:
                        continue
            
            # Sort by timestamp (oldest first)
            backup_dirs.sort(key=lambda x: x[1])
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - self._retention_policy
            
            # Delete backups older than cutoff
            for backup_dir, timestamp in backup_dirs:
                if timestamp < cutoff_date:
                    import shutil
                    shutil.rmtree(backup_dir)
                    logger.info(f"Deleted knowledge backup {backup_dir} (older than retention policy)")
        except Exception as e:
            logger.error(f"Error applying retention policy: {e}")

class ResourceUsageTracker:
    """Tracks resource usage for operations to support sustainability metrics."""
    
    def __init__(self):
        self.operation_stats = defaultdict(lambda: {
            "count": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_time": 0,
            "total_bytes": 0,
            "energy_estimated": 0
        })
        self.active_operations = {}
        self.start_time = time.time()
    
    def start_tracking(self, operation_type: str):
        """Start tracking a resource-intensive operation."""
        self.active_operations[operation_type] = {
            "start_time": time.time(),
            "start_cpu": psutil.cpu_percent(interval=None),
            "start_memory": psutil.virtual_memory().percent
        }
    
    def end_tracking(self, operation_type: str, success: bool, bytes_processed: int = 0):
        """End tracking a resource-intensive operation."""
        if operation_type not in self.active_operations:
            return
        
        op_data = self.active_operations.pop(operation_type)
        duration = time.time() - op_data["start_time"]
        
        # Update statistics
        stats = self.operation_stats[operation_type]
        stats["count"] += 1
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
        stats["total_time"] += duration
        stats["total_bytes"] += bytes_processed
        
        # Estimate energy usage (simplified)
        # In a real implementation, this would use more sophisticated models
        cpu_usage = psutil.cpu_percent(interval=None) - op_data["start_cpu"]
        memory_usage = psutil.virtual_memory().percent - op_data["start_memory"]
        energy_estimate = (cpu_usage * 0.1 + memory_usage * 0.05) * duration
        stats["energy_estimated"] += energy_estimate
        
        # Log for observability
        logger.debug(f"Operation {operation_type} completed in {duration:.2f}s. Success: {success}. Energy: {energy_estimate:.4f} kWh")
    
    def track_operation(self, operation_type: str, bytes_processed: int = 0):
        """Track a completed operation without start/end tracking."""
        stats = self.operation_stats[operation_type]
        stats["count"] += 1
        stats["success_count"] += 1
        stats["total_bytes"] += bytes_processed
    
    def get_stats(self) -> dict:
        """Get resource usage statistics."""
        return {
            "total_runtime": time.time() - self.start_time,
            "operations": dict(self.operation_stats)
        }

#######################
# GROWTH ANALYZER (Growth & Abundance)
#######################
class GrowthAnalyzer:
    """Analyzes node performance, learns, and recommends optimizations with comprehensive metrics."""
    
    def __init__(self, node):
        self.node = node
        self.growth_file = os.path.join(node.node_dir, "growth", "growth.json")
        self._last_resource_check = time.time()
        self._resource_check_interval = 5  # seconds
        self._system_metrics = {
            "cpu_percent": [],
            "memory_percent": [],
            "disk_percent": []
        }
        self._max_metrics_history = 100
        self._energy_efficiency = 0.0
        self._carbon_footprint = 0.0
        self._last_energy_calculation = time.time()
        self._energy_calculation_interval = 60  # seconds
        
        # Initialize RL agent
        self.rl_agent = self.node.knowledge_processor.rl_agent
    
    def analyze(self):
        """Analyze the node's performance and update growth metrics with comprehensive system monitoring."""
        try:
            # Check system resources periodically
            self._check_system_resources()
            
            # Load current growth data
            growth_data = self._load_growth_data()
            
            # Analyze knowledge
            knowledge_count = len(os.listdir(os.path.join(self.node.node_dir, "knowledge")))
            
            # Analyze communication
            inbox_dir = os.path.join(self.node.node_dir, "communication", "inbox")
            outbox_dir = os.path.join(self.node.node_dir, "communication", "outbox")
            inbox_count = len(os.listdir(inbox_dir))
            outbox_count = len(os.listdir(outbox_dir))
            
            # Analyze network
            known_nodes_count = len(self.node.known_nodes)
            
            # Analyze processing load
            processing_load = self._get_processing_load()
            
            # Calculate principles alignment
            principles_alignment = self.node.principles_engine.evaluate_code_against_principles(__file__).get("overall", 0)
            
            # Update metrics
            timestamp = time.time()
            if "history" not in growth_data:
                growth_data["history"] = []
            
            growth_data["history"].append({
                "timestamp": timestamp,
                "knowledge_count": knowledge_count,
                "inbox_count": inbox_count,
                "outbox_count": outbox_count,
                "known_nodes_count": known_nodes_count,
                "processing_load": processing_load,
                "principles_alignment": principles_alignment,
                "cpu_percent": self._system_metrics["cpu_percent"][-1] if self._system_metrics["cpu_percent"] else 0,
                "memory_percent": self._system_metrics["memory_percent"][-1] if self._system_metrics["memory_percent"] else 0,
                "disk_percent": self._system_metrics["disk_percent"][-1] if self._system_metrics["disk_percent"] else 0,
                "energy_efficiency": self._energy_efficiency,
                "carbon_footprint": self._carbon_footprint
            })
            
            # Limit history size
            if len(growth_data["history"]) > 200:
                growth_data["history"] = growth_data["history"][-200:]
            
            # Calculate trends and generate insights
            insights = growth_data.get("insights", [])
            if len(growth_data["history"]) > 1:
                prev = growth_data["history"][-2]
                current = growth_data["history"][-1]
                
                # Knowledge growth
                knowledge_growth = current["knowledge_count"] - prev["knowledge_count"]
                if knowledge_growth > 0:
                    insights.append({
                        "timestamp": timestamp,
                        "type": "knowledge_growth",
                        "message": f"Knowledge base grew by {knowledge_growth} items",
                        "value": knowledge_growth
                    })
                
                # Network changes
                network_change = current["known_nodes_count"] - prev["known_nodes_count"]
                if network_change != 0:
                    insights.append({
                        "timestamp": timestamp,
                        "type": "network_change",
                        "message": f"Network {'grew' if network_change > 0 else 'shrunk'} by {abs(network_change)} nodes",
                        "value": network_change
                    })
                
                # Inbox pressure
                inbox_change = current["inbox_count"] - prev["inbox_count"]
                if inbox_change > 10:
                    insights.append({
                        "timestamp": timestamp,
                        "type": "inbox_pressure",
                        "message": f"Inbox grew significantly by {inbox_change} messages",
                        "value": inbox_change
                    })
                
                # Principles alignment changes
                principles_change = current["principles_alignment"] - prev["principles_alignment"]
                if principles_change > 0.1:
                    insights.append({
                        "timestamp": timestamp,
                        "type": "principles_alignment_growth",
                        "message": f"Principles alignment improved by {principles_change:.2f}",
                        "value": principles_change
                    })
                elif principles_change < -0.1:
                    insights.append({
                        "timestamp": timestamp,
                        "type": "principles_alignment_decline",
                        "message": f"Principles alignment declined by {abs(principles_change):.2f}",
                        "value": principles_change
                    })
                
                # Resource usage trends
                resource_threshold = 0.8  # 80% usage
                if current["cpu_percent"] > resource_threshold:
                    insights.append({
                        "timestamp": timestamp,
                        "type": "resource_pressure",
                        "message": f"CPU usage at {current['cpu_percent']:.1f}% (approaching limit)",
                        "value": current["cpu_percent"]
                    })
                if current["memory_percent"] > resource_threshold:
                    insights.append({
                        "timestamp": timestamp,
                        "type": "resource_pressure",
                        "message": f"Memory usage at {current['memory_percent']:.1f}% (approaching limit)",
                        "value": current["memory_percent"]
                    })
                if current["disk_percent"] > resource_threshold:
                    insights.append({
                        "timestamp": timestamp,
                        "type": "resource_pressure",
                        "message": f"Disk usage at {current['disk_percent']:.1f}% (approaching limit)",
                        "value": current["disk_percent"]
                    })
                
                # Sustainability metrics
                if current["energy_efficiency"] < CONFIG["sustainability"]["energy_efficiency_target"]:
                    insights.append({
                        "timestamp": timestamp,
                        "type": "sustainability",
                        "message": f"Energy efficiency at {current['energy_efficiency']:.2f} (below target of {CONFIG['sustainability']['energy_efficiency_target']})",
                        "value": current["energy_efficiency"]
                    })
            
            # Limit insights
            if len(insights) > 100:
                growth_data["insights"] = insights[-100:]
            else:
                growth_data["insights"] = insights
            
            # Update last_updated timestamp
            growth_data["last_updated"] = timestamp
            
            # Save updated growth data
            self._save_growth_data(growth_data)
            
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                NETWORK_NODES.labels(node_id=self.node.id).set(known_nodes_count)
                PRINCIPLES_ALIGNMENT.labels(node_id=self.node.id, principle="overall").set(principles_alignment)
                CPU_USAGE.labels(node_id=self.node.id).set(self._system_metrics["cpu_percent"][-1] if self._system_metrics["cpu_percent"] else 0)
                MEMORY_USAGE.labels(node_id=self.node.id).set(self._system_metrics["memory_percent"][-1] if self._system_metrics["memory_percent"] else 0)
                DISK_USAGE.labels(node_id=self.node.id).set(self._system_metrics["disk_percent"][-1] if self._system_metrics["disk_percent"] else 0)
            
            logger.info(f"Node {self.node.id} updated growth metrics. Principles Alignment: {growth_data['history'][-1]['principles_alignment']:.2f}")
            
            # Trigger RL optimization recommendation
            self.recommend_optimization()
        except Exception as e:
            logger.error(f"Error analyzing growth: {e}")
            logger.error(traceback.format_exc())
    
    def _check_system_resources(self):
        """Check system resource usage and track for sustainability metrics."""
        current_time = time.time()
        if current_time - self._last_resource_check < self._resource_check_interval:
            return
        
        # Get current resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage(BASE_DIR).percent
        
        # Store in history
        self._system_metrics["cpu_percent"].append(cpu_percent)
        self._system_metrics["memory_percent"].append(memory_percent)
        self._system_metrics["disk_percent"].append(disk_percent)
        
        # Limit history size
        for metric in self._system_metrics:
            if len(self._system_metrics[metric]) > self._max_metrics_history:
                self._system_metrics[metric] = self._system_metrics[metric][-self._max_metrics_history:]
        
        # Calculate energy efficiency
        self._calculate_energy_efficiency()
        
        # Update carbon footprint if tracking
        if self.node.knowledge_processor.carbon_tracker and current_time - self._last_energy_calculation > self._energy_calculation_interval:
            self._update_carbon_footprint()
            self._last_energy_calculation = current_time
        
        self._last_resource_check = current_time
    
    def _get_processing_load(self) -> float:
        """Get a normalized processing load value."""
        # This could be based on queue length, CPU usage, etc.
        inbox_dir = os.path.join(self.node.node_dir, "communication", "inbox")
        inbox_count = len(os.listdir(inbox_dir))
        max_inbox = CONFIG["node"]["max_inbox_size"]
        return min(1.0, inbox_count / max_inbox) if max_inbox > 0 else 0
    
    def _calculate_energy_efficiency(self):
        """Calculate energy efficiency based on resource usage and work done."""
        # Simplified energy efficiency calculation
        # In a real implementation, this would use more sophisticated models
        if not self._system_metrics["cpu_percent"]:
            return
        
        # Get average resource usage over the last few samples
        avg_cpu = sum(self._system_metrics["cpu_percent"][-5:]) / min(5, len(self._system_metrics["cpu_percent"]))
        avg_memory = sum(self._system_metrics["memory_percent"][-5:]) / min(5, len(self._system_metrics["memory_percent"]))
        
        # Get work done (simplified - could be based on message processing, knowledge updates, etc.)
        work_done = len(self.node.knowledge_processor.resource_tracker.operation_stats.get("store", {}).get("count", 0))
        
        # Calculate efficiency (higher is better)
        if avg_cpu > 0:
            self._energy_efficiency = work_done / avg_cpu
        else:
            self._energy_efficiency = 0.0
    
    def _update_carbon_footprint(self):
        """Update carbon footprint metrics if tracking is enabled."""
        if not self.node.knowledge_processor.carbon_tracker:
            return
        
        try:
            # Get current emissions
            emissions = self.node.knowledge_processor.carbon_tracker.flush()
            if emissions:
                self._carbon_footprint = emissions.get('emissions', 0.0)
                logger.debug(f"Carbon footprint: {self._carbon_footprint:.6f} kgCO2e")
        except Exception as e:
            logger.error(f"Error updating carbon footprint: {e}")
    
    def get_growth_data(self):
        """Get the current growth data."""
        return self._load_growth_data()
    
    def _load_growth_data(self):
        """Safely load growth data."""
        try:
            if os.path.exists(self.growth_file):
                with open(self.growth_file, 'r') as f:
                    return json.load(f)
            return {"metrics": {}, "insights": [], "history": [], "last_updated": time.time()}
        except Exception as e:
            logger.error(f"Error loading growth data from {self.growth_file}: {e}")
            return {"metrics": {}, "insights": [], "history": [], "last_updated": time.time()}
    
    def _save_growth_data(self, growth_data):
        """Safely save growth data."""
        try:
            with open(self.growth_file, 'w') as f:
                json.dump(growth_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving growth data to {self.growth_file}: {e})

#######################
# CONSCIOUSNESS STREAM (Love & Clarity)
#######################
class ConsciousnessStream:
    """Synthesizes node state, events, and insights into a coherent stream with comprehensive logging."""
    
    def __init__(self, node):
        self.node = node
        self._stream = []
        self._max_stream_size = 200
        self._log_file = os.path.join(node.node_dir, "logs", "consciousness.log")
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
        self._log_lock = RLock()
        self._last_log_rotation = time.time()
        self._log_rotation_interval = 86400  # Rotate daily
        self._log_retention_days = CONFIG["observability"]["log_retention_days"]
    
    def add_event(self, event_type, message, details=None):
        """Add an event to the consciousness stream with comprehensive logging."""
        timestamp = time.time()
        stream_entry = {
            "timestamp": timestamp,
            "node_id": self.node.id,
            "type": event_type,
            "message": message,
            "details": details
        }
        
        # Add to in-memory stream
        self._stream.append(stream_entry)
        if len(self._stream) > self._max_stream_size:
            self._stream = self._stream[-self._max_stream_size:]
        
        # Write to log file
        self._write_to_log(stream_entry)
        
        # Check if we need to rotate logs
        self._rotate_logs_if_needed()
    
    def _write_to_log(self, entry):
        """Write an entry to the consciousness log file."""
        with self._log_lock:
            try:
                # Format the log entry
                log_time = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S.%f")
                log_entry = f"[{log_time}] [{entry['type']}] Node {entry['node_id']}: {entry['message']}"
                if entry["details"]:
                    log_entry += f" | Details: {json.dumps(entry['details'])}"
                
                # Write to file
                with open(self._log_file, 'a') as f:
                    f.write(log_entry + "\n")
            except Exception as e:
                logger.error(f"Error writing to consciousness log: {e}")
    
    def _rotate_logs_if_needed(self):
        """Rotate log files if needed based on retention policy."""
        current_time = time.time()
        if current_time - self._last_log_rotation < self._log_rotation_interval:
            return
        
        try:
            # Create a timestamped backup of the current log
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self._log_file}.{timestamp}"
            import shutil
            shutil.copy2(self._log_file, backup_file)
            
            # Clear the current log file
            open(self._log_file, 'w').close()
            
            # Clean up old log backups
            self._cleanup_old_logs()
            
            self._last_log_rotation = current_time
            logger.info(f"Rotated consciousness log to {backup_file}")
        except Exception as e:
            logger.error(f"Error rotating consciousness log: {e}")
    
    def _cleanup_old_logs(self):
        """Clean up old log files according to retention policy."""
        try:
            # Get all log files
            log_files = []
            log_dir = os.path.dirname(self._log_file)
            log_prefix = os.path.basename(self._log_file)
            
            for filename in os.listdir(log_dir):
                if filename.startswith(log_prefix) and '.' in filename:
                    parts = filename.split('.')
                    if len(parts) > 1 and len(parts[-1]) == 14:  # Timestamp format YYYYMMDD_HHMMSS
                        try:
                            timestamp = datetime.strptime(parts[-1], "%Y%m%d_%H%M%S")
                            log_files.append((os.path.join(log_dir, filename), timestamp))
                        except ValueError:
                            continue
            
            # Sort by timestamp (oldest first)
            log_files.sort(key=lambda x: x[1])
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self._log_retention_days)
            
            # Delete logs older than cutoff
            for log_file, timestamp in log_files:
                if timestamp < cutoff_date:
                    os.remove(log_file)
                    logger.info(f"Deleted old consciousness log: {log_file}")
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
    
    def get_stream(self, limit=50):
        """Get the latest entries from the consciousness stream."""
        return self._stream[-limit:]
    
    def synthesize_summary(self, recent_events=None):
        """Synthesize a summary from recent stream events and node state with AI enhancement."""
        if recent_events is None:
            recent_events = self.get_stream(limit=20)
        
        summary = f"Consciousness Stream Summary for Node {self.node.id} ({datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}):\n"
        summary += f"  - State: Alive, Role: {self.node.role}, Capabilities: {', '.join(self.node.capabilities)}\n"
        summary += f"  - Known Nodes: {len(self.node.known_nodes)}, Inbox Size: {len(os.listdir(os.path.join(self.node.node_dir, 'communication', 'inbox')))}\n"
        
        growth_data = self.node.growth_analyzer.get_growth_data()
        if growth_data and growth_data["history"]:
            latest_metrics = growth_data["history"][-1]
            summary += f"  - Knowledge Count: {latest_metrics.get('knowledge_count', 0)}, Principles Alignment: {latest_metrics.get('principles_alignment', 0):.2f}\n"
            if growth_data["insights"]:
                latest_insight = growth_data["insights"][-1]
                summary += f"  - Latest Insight: {latest_insight['message']} ({datetime.fromtimestamp(latest_insight['timestamp']).strftime('%H:%M')})\n"
        
        if recent_events:
            summary += "  - Recent Events:\n"
            for event in recent_events:
                summary += f"    - [{datetime.fromtimestamp(event['timestamp']).strftime('%H:%M')}] {event['type']}: {event['message']}\n"
        else:
            summary += "  - No recent significant events.\n"
        
        # Add system resource information
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage(BASE_DIR).percent
        summary += f"  - System Resources: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%\n"
        
        # Add sustainability metrics
        if CONFIG["sustainability"]["carbon_footprint_tracking"]:
            summary += f"  - Carbon Footprint: {self.node.growth_analyzer._carbon_footprint:.6f} kgCO2e\n"
            summary += f"  - Energy Efficiency: {self.node.growth_analyzer._energy_efficiency:.2f}\n"
        
        # Use AI reasoner to synthesize a more nuanced summary if available
        if self.node.knowledge_processor.reasoner:
            try:
                ai_summary_prompt = f"Synthesize the following information into a concise, insightful summary about the state and recent activity of a network node:\n{summary}\nAI Synthesis:"
                ai_response = self.node.knowledge_processor.reasoner(ai_summary_prompt, max_length=300, num_return_sequences=1, truncation=True)[0]["generated_text"]
                ai_synthesis = ai_response.split("AI Synthesis:")[-1].strip()
                summary += f"\nAI Synthesis:\n{ai_synthesis}\n"
            except Exception as e:
                logger.error(f"Error during AI synthesis: {e}")
                summary += "\nAI synthesis currently unavailable.\n"
        
        # Add deployment information
        summary += f"\nDeployment Information:\n"
        summary += f"  - Environment: {CONFIG['deployment']['environment']}\n"
        summary += f"  - Version: {CONFIG['deployment']['version']}\n"
        summary += f"  - Build ID: {CONFIG['deployment']['build_id']}\n"
        summary += f"  - Hostname: {CONFIG['system']['hostname']}\n"
        summary += f"  - OS: {CONFIG['system']['os']} {CONFIG['system']['os_version']}\n"
        
        return summary

#######################
# RELATIONSHIP NURTURING (Love & Growth)
#######################
class RelationshipNurturer:
    """Manages and nurtures relationships with other nodes with comprehensive relationship management."""
    
    RELATIONSHIP_TYPES = ["symbiotic", "mentoring", "collaborative", "observer"]
    RELATIONSHIP_STATES = ["active", "pending", "inactive", "terminated"]
    
    def __init__(self, node):
        self.node = node
        self.relationships = {}  # {node_id: {type: type, status: 'active'/'pending'/'inactive', started_at: timestamp, metrics: {}}}
        self._relationship_history = []
        self._max_history_size = 1000
        self._trust_scores = {}  # {node_id: trust_score}
        self._last_trust_update = time.time()
        self._trust_update_interval = 300  # seconds
    
    def propose_relationship(self, target_node_id, relationship_type, purpose="general"):
        """Propose a relationship to another node with trust-based validation."""
        if relationship_type not in self.RELATIONSHIP_TYPES:
            logger.warning(f"Invalid relationship type: {relationship_type}")
            return False
        
        if target_node_id == self.node.id:
            logger.warning("Cannot form relationship with self.")
            return False
        
        # Check trust score before proposing
        if target_node_id in self._trust_scores and self._trust_scores[target_node_id] < 0.3:
            logger.warning(f"Trust score too low ({self._trust_scores[target_node_id]}) to propose relationship with {target_node_id}")
            return False
        
        if target_node_id in self.relationships and self.relationships[target_node_id]["status"] == "pending":
            logger.info(f"Relationship proposal to {target_node_id} already pending.")
            return False
        
        message = {
            "type": "relationship_proposal",
            "source": self.node.id,
            "target": target_node_id,
            "relationship_type": relationship_type,
            "purpose": purpose,
            "vector_clock": self.node.increment_vector_clock(),
            "trust_score": self._calculate_trust_score(self.node.id)
        }
        
        if self.node._send_message_to_node(target_node_id, message):
            self.relationships[target_node_id] = {
                "type": relationship_type,
                "status": "pending",
                "started_at": time.time(),
                "purpose": purpose,
                "metrics": {
                    "proposals_sent": 1,
                    "responses_received": 0,
                    "trust_score": self._calculate_trust_score(target_node_id)
                }
            }
            
            # Record in history
            self._record_relationship_event("proposal_sent", target_node_id, {
                "relationship_type": relationship_type,
                "purpose": purpose
            })
            
            logger.info(f"Proposed {relationship_type} relationship to {target_node_id}")
            self.node.consciousness_stream.add_event("relationship_proposal", f"Proposed {relationship_type} relationship to {target_node_id}")
            return True
        else:
            logger.warning(f"Failed to send relationship proposal to {target_node_id}")
            self.node.consciousness_stream.add_event("relationship_proposal_failed", f"Failed to propose {relationship_type} relationship to {target_node_id}")
            return False
    
    def accept_relationship(self, proposing_node_id, relationship_type, purpose="general"):
        """Accept a relationship proposal from another node with trust validation."""
        # Validate trust score of proposing node
        trust_score = self._calculate_trust_score(proposing_node_id)
        if trust_score < 0.2:
            logger.warning(f"Trust score too low ({trust_score}) to accept relationship from {proposing_node_id}")
            return False
        
        if proposing_node_id in self.relationships and self.relationships[proposing_node_id]["status"] == "active":
            logger.info(f"Relationship with {proposing_node_id} already active.")
            return True
        
        self.relationships[proposing_node_id] = {
            "type": relationship_type,
            "status": "active",
            "started_at": time.time(),
            "purpose": purpose,
            "metrics": {
                "proposals_sent": 0,
                "responses_received": 1,
                "trust_score": trust_score,
                "last_interaction": time.time(),
                "interaction_count": 1
            }
        }
        
        # Record in history
        self._record_relationship_event("accepted", proposing_node_id, {
            "relationship_type": relationship_type,
            "purpose": purpose
        })
        
        logger.info(f"Accepted {relationship_type} relationship from {proposing_node_id}")
        self.node.consciousness_stream.add_event("relationship_accepted", f"Accepted {relationship_type} relationship from {proposing_node_id}")
        
        # Send acceptance message back
        message = {
            "type": "relationship_accepted",
            "source": self.node.id,
            "target": proposing_node_id,
            "relationship_type": relationship_type,
            "vector_clock": self.node.increment_vector_clock(),
            "trust_score": self._calculate_trust_score(self.node.id)
        }
        self.node._send_message_to_node(proposing_node_id, message)
        
        # Update trust score
        self._update_trust_score(proposing_node_id, 0.1)
        
        return True
    
    def _record_relationship_event(self, event_type, node_id, details=None):
        """Record a relationship event in the history."""
        event = {
            "timestamp": time.time(),
            "node_id": self.node.id,
            "event_type": event_type,
            "target_node_id": node_id,
            "details": details or {}
        }
        
        self._relationship_history.append(event)
        if len(self._relationship_history) > self._max_history_size:
            self._relationship_history = self._relationship_history[-self._max_history_size:]
    
    def get_relationship_history(self, limit=100):
        """Get the relationship history."""
        return self._relationship_history[-limit:]
    
    def _calculate_trust_score(self, node_id) -> float:
        """Calculate a trust score for a node based on various factors."""
        # Base trust score
        trust = 0.5
        
        # If we have existing relationship metrics
        if node_id in self.relationships:
            metrics = self.relationships[node_id]["metrics"]
            trust = metrics.get("trust_score", 0.5)
            
            # Adjust based on interaction count
            interaction_count = metrics.get("interaction_count", 0)
            trust += min(0.2, interaction_count * 0.01)
            
            # Adjust based on recent interactions
            last_interaction = metrics.get("last_interaction", 0)
            time_since_interaction = time.time() - last_interaction
            if time_since_interaction < 3600:  # Less than an hour
                trust += 0.1
            elif time_since_interaction < 86400:  # Less than a day
                trust += 0.05
        
        # Adjust based on principles alignment (if known)
        if node_id in self.node.known_nodes:
            alignment = self.node.known_nodes[node_id].get("principles_alignment", 0)
            trust = trust * 0.7 + alignment * 0.3
        
        # Cap between 0 and 1
        return max(0, min(1, trust))
    
    def _update_trust_score(self, node_id, delta):
        """Update the trust score for a node."""
        current = self._trust_scores.get(node_id, 0.5)
        new_score = max(0, min(1, current + delta))
        self._trust_scores[node_id] = new_score
        
        # Also update in relationship metrics if exists
        if node_id in self.relationships:
            self.relationships[node_id]["metrics"]["trust_score"] = new_score
    
    def nurture(self):
        """Periodically review and nurture existing relationships with comprehensive management."""
        logger.debug(f"Node {self.node.id} nurturing relationships.")
        
        # Update trust scores periodically
        self._update_trust_scores()
        
        # Identify potential partners
        potential_partners = [node_id for node_id in self.node.known_nodes 
                             if node_id not in self.relationships or self.relationships[node_id]["status"] != "active"]
        
        # Prioritize nurturing existing relationships
        for partner_id, relationship_info in list(self.relationships.items()):
            if relationship_info["status"] == "active":
                self._evaluate_relationship_health(partner_id, relationship_info)
                
                # Send a check-in message or share relevant knowledge
                if random.random() < 0.2:
                    checkin_message = {
                        "type": "check_in",
                        "source": self.node.id,
                        "target": partner_id,
                        "vector_clock": self.node.increment_vector_clock(),
                        "message": f"Checking in, partner. Node {self.node.id} is operational.",
                        "trust_score": self._calculate_trust_score(self.node.id)
                    }
                    self.node._send_message_to_node(partner_id, checkin_message)
                    self._update_trust_score(partner_id, 0.01)  # Small positive update for interaction
        
        # Propose new relationships based on strategy
        if random.random() < 0.1 and potential_partners:
            target_node_id = random.choice(potential_partners)
            target_info = self.node.known_nodes.get(target_node_id, {})
            
            # Determine relationship type based on potential partner's capabilities/role
            if "knowledge" in target_info.get("capabilities", []):
                rel_type = "collaborative"
            elif target_info.get("role") == "mentor":
                rel_type = "mentoring"
            else:
                rel_type = random.choice(["symbiotic", "observer"])
            
            # Only propose if trust score is sufficient
            if self._calculate_trust_score(target_node_id) > 0.3:
                self.propose_relationship(target_node_id, rel_type)
    
    def _update_trust_scores(self):
        """Update trust scores for all known nodes."""
        current_time = time.time()
        if current_time - self._last_trust_update < self._trust_update_interval:
            return
        
        # Update trust scores based on recent interactions
        for node_id in self.node.known_nodes:
            # Gradual decay of trust over time without interaction
            if node_id in self.relationships:
                last_interaction = self.relationships[node_id]["metrics"].get("last_interaction", 0)
                time_since_interaction = current_time - last_interaction
                decay = min(0.05, time_since_interaction / 86400 * 0.01)  # Max 5% decay per day
                self._update_trust_score(node_id, -decay)
        
        self._last_trust_update = current_time
    
    def _evaluate_relationship_health(self, partner_id, relationship_info):
        """Evaluate the health of an active relationship with comprehensive metrics."""
        now = time.time()
        last_seen = self.node.known_nodes.get(partner_id, {}).get("last_seen", 0)
        
        # Update interaction metrics
        relationship_info["metrics"]["last_interaction"] = now
        relationship_info["metrics"]["interaction_count"] = relationship_info["metrics"].get("interaction_count", 0) + 1
        
        # Check for inactivity
        if now - last_seen > CONFIG["node"]["cleanup_interval"] * 2:
            logger.warning(f"Relationship with {partner_id} seems inactive. Marking as inactive.")
            relationship_info["status"] = "inactive"
            self.node.consciousness_stream.add_event("relationship_inactive", f"Relationship with {partner_id} became inactive.")
            
            # Record in history
            self._record_relationship_event("inactive", partner_id, {
                "reason": "inactivity",
                "last_seen": last_seen
            })
        
        # Evaluate trust score
        trust_score = self._calculate_trust_score(partner_id)
        if trust_score < 0.2:
            logger.warning(f"Relationship with {partner_id} has low trust score ({trust_score}). Considering termination.")
            
            # Record in history
            self._record_relationship_event("low_trust", partner_id, {
                "trust_score": trust_score
            })
            
            # Consider terminating if trust remains low
            if trust_score < 0.1:
                self.terminate_relationship(partner_id, "low_trust")
    
    def terminate_relationship(self, partner_id, reason="unknown"):
        """Terminate a relationship with proper notification."""
        if partner_id not in self.relationships:
            return False
        
        # Send termination notification
        message = {
            "type": "relationship_terminated",
            "source": self.node.id,
            "target": partner_id,
            "reason": reason,
            "vector_clock": self.node.increment_vector_clock()
        }
        self.node._send_message_to_node(partner_id, message)
        
        # Update relationship status
        self.relationships[partner_id]["status"] = "terminated"
        self.relationships[partner_id]["terminated_at"] = time.time()
        self.relationships[partner_id]["termination_reason"] = reason
        
        # Record in history
        self._record_relationship_event("terminated", partner_id, {
            "reason": reason
        })
        
        logger.info(f"Terminated relationship with {partner_id} due to {reason}")
        self.node.consciousness_stream.add_event("relationship_terminated", f"Terminated relationship with {partner_id} due to {reason}")
        
        return True

#######################
# SELF-HEALING (Safety & Growth)
#######################
class SelfHealing:
    """Monitors for anomalies and initiates self-repair or adaptation with comprehensive recovery strategies."""
    
    def __init__(self, node):
        self.node = node
        self._anomaly_threshold = 0.9
        self._recovery_strategies = {
            "low_principles_alignment": self._recover_from_low_alignment,
            "inbox_pressure": self._recover_from_inbox_pressure,
            "resource_overload": self._recover_from_resource_overload,
            "network_isolation": self._recover_from_network_isolation,
            "data_corruption": self._recover_from_data_corruption
        }
        self._recovery_history = []
        self._max_recovery_history = 100
        self._last_recovery_time = 0
        self._min_recovery_interval = 60  # seconds
    
    def monitor_anomalies(self):
        """Monitor node state and metrics for anomalies with comprehensive checks."""
        # Check principles alignment
        growth_data = self.node.growth_analyzer.get_growth_data()
        if growth_data and growth_data["history"]:
            latest_alignment = growth_data["history"][-1].get("principles_alignment", 0)
            if latest_alignment < self._anomaly_threshold:
                logger.warning(f"Principles alignment ({latest_alignment:.2f}) below threshold ({self._anomaly_threshold}). Initiating self-healing assessment.")
                self.assess_and_heal("low_principles_alignment", {"current_alignment": latest_alignment})
        
        # Check inbox size
        inbox_dir = os.path.join(self.node.node_dir, "communication", "inbox")
        inbox_count = len(os.listdir(inbox_dir))
        if inbox_count > CONFIG["node"]["max_inbox_size"] * 0.9:
            logger.warning(f"Inbox nearing capacity ({inbox_count}/{CONFIG['node']['max_inbox_size']}). Initiating self-healing assessment.")
            self.assess_and_heal("inbox_pressure", {"inbox_count": inbox_count})
        
        # Check resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage(BASE_DIR).percent
        
        resource_thresholds = CONFIG["node"]["resource_limits"]
        if (cpu_percent > resource_thresholds["cpu_percent"] or
            memory_percent > resource_thresholds["memory_percent"] or
            disk_percent > resource_thresholds["disk_percent"]):
            logger.warning(f"Resource overload detected (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%). Initiating self-healing assessment.")
            self.assess_and_heal("resource_overload", {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent
            })
        
        # Check network connectivity
        if len(self.node.known_nodes) < 2 and time.time() - self.node.last_heartbeat > CONFIG["node"]["heartbeat_interval"] * 3:
            logger.warning(f"Potential network isolation detected (known nodes: {len(self.node.known_nodes)}). Initiating self-healing assessment.")
            self.assess_and_heal("network_isolation", {"known_nodes": len(self.node.known_nodes)})
    
    def assess_and_heal(self, anomaly_type, anomaly_details=None):
        """Assess the anomaly and determine healing strategy with risk assessment."""
        current_time = time.time()
        
        # Rate limit recovery attempts
        if current_time - self._last_recovery_time < self._min_recovery_interval:
            logger.debug(f"Recovery attempt for {anomaly_type} skipped due to rate limiting")
            return
        
        logger.info(f"Assessing anomaly: {anomaly_type}")
        self.node.consciousness_stream.add_event("anomaly_detected", anomaly_type, anomaly_details)
        
        # Risk assessment before recovery
        risk_level = self._assess_recovery_risk(anomaly_type, anomaly_details)
        if risk_level == "high":
            logger.warning(f"Recovery for {anomaly_type} has high risk. Proceeding with caution.")
            # Might require human intervention in production
        elif risk_level == "critical":
            logger.critical(f"Recovery for {anomaly_type} has critical risk. Aborting automatic recovery.")
            return
        
        # Determine healing strategy
        strategy = self._recovery_strategies.get(anomaly_type)
        if not strategy:
            logger.warning(f"No recovery strategy for anomaly type: {anomaly_type}")
            return
        
        # Execute recovery
        logger.info(f"Initiating healing strategy for {anomaly_type}")
        self._last_recovery_time = current_time
        try:
            strategy(anomaly_details)
            self._record_recovery(anomaly_type, anomaly_details, "success")
        except Exception as e:
            logger.error(f"Recovery failed for {anomaly_type}: {e}")
            self._record_recovery(anomaly_type, anomaly_details, "failure", str(e))
    
    def _assess_recovery_risk(self, anomaly_type, anomaly_details) -> str:
        """Assess the risk level of a recovery operation."""
        # Default risk level
        risk = "medium"
        
        # Special cases for high-risk recoveries
        if anomaly_type == "data_corruption":
            risk = "high"
        elif anomaly_type == "resource_overload" and anomaly_details.get("disk_percent", 0) > 95:
            risk = "critical"
        
        return risk
    
    def _record_recovery(self, anomaly_type, details, status, error=None):
        """Record a recovery attempt in the history."""
        recovery = {
            "timestamp": time.time(),
            "anomaly_type": anomaly_type,
            "details": details,
            "status": status,
            "error": error
        }
        
        self._recovery_history.append(recovery)
        if len(self._recovery_history) > self._max_recovery_history:
            self._recovery_history = self._recovery_history[-self._max_recovery_history:]
    
    def get_recovery_history(self, limit=50):
        """Get the recovery history."""
        return self._recovery_history[-limit:]
    
    def _recover_from_low_alignment(self, details):
        """Recover from low principles alignment."""
        growth_data = self.node.growth_analyzer.get_growth_data()
        if growth_data and growth_data["history"]:
            latest_metrics = growth_data["history"][-1]
            principle_focus = self.node.principles_engine.recommend_evolution_focus({"average": latest_metrics.get("principles_alignment", {})})
            
            if principle_focus:
                logger.info(f"Triggering principles-focused evolution on: {principle_focus}")
                self.node.consciousness_stream.add_event("principles_focused_evolution", f"Focusing on {principle_focus}")
                
                # Instead of directly modifying code, create a safe evolution plan
                self._create_evolution_plan(principle_focus)
    
    def _create_evolution_plan(self, principle_focus):
        """Create a safe evolution plan for principle-focused improvement."""
        # Create a plan directory
        plan_dir = os.path.join(self.node.node_dir, "evolution_plans", f"plan_{int(time.time())}")
        os.makedirs(plan_dir, exist_ok=True)
        
        # Create a plan manifest
        manifest = {
            "timestamp": time.time(),
            "principle_focus": principle_focus,
            "status": "proposed",
            "changes": [],
            "backup_required": True,
            "rollback_plan": []
        }
        
        # For demonstration, we'll plan to add a comment (in real system, this would be more complex)
        manifest["changes"].append({
            "file": __file__,
            "line_range": [10, 10],
            "before": "",
            "after": f"# ✨ Enhanced for {principle_focus.upper()} ✨\n"
        })
        
        # Create a rollback plan (revert the change)
        manifest["rollback_plan"].append({
            "file": __file__,
            "line_range": [10, 11],  # Including the new line
            "before": manifest["changes"][0]["after"],
            "after": manifest["changes"][0]["before"]
        })
        
        # Save the plan
        with open(os.path.join(plan_dir, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create a backup if needed
        if manifest["backup_required"]:
            backup_dir = os.path.join(plan_dir, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            import shutil
            shutil.copy2(__file__, os.path.join(backup_dir, os.path.basename(__file__)))
        
        logger.info(f"Created evolution plan for {principle_focus} at {plan_dir}")
        
        # Execute the plan with safety checks
        self._execute_evolution_plan(plan_dir)
    
    def _execute_evolution_plan(self, plan_dir):
        """Execute an evolution plan with comprehensive safety checks."""
        # Load the plan manifest
        manifest_path = os.path.join(plan_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            logger.error(f"Evolution plan manifest not found at {manifest_path}")
            return False
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Check preconditions
        if not self._check_preconditions(manifest):
            logger.error("Evolution plan preconditions not met")
            return False
        
        try:
            # Apply changes
            for change in manifest["changes"]:
                self._apply_change(change)
            
            # Update manifest status
            manifest["status"] = "applied"
            with open(os.path.join(plan_dir, "manifest.json"), 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Verify post-conditions
            if not self._verify_postconditions(manifest):
                logger.warning("Post-conditions not met after evolution. Rolling back.")
                self._rollback_evolution_plan(plan_dir)
                return False
            
            logger.info("Evolution plan successfully applied")
            return True
        except Exception as e:
            logger.error(f"Error applying evolution plan: {e}")
            self._rollback_evolution_plan(plan_dir)
            return False
    
    def _check_preconditions(self, manifest) -> bool:
        """Check if preconditions for an evolution plan are met."""
        # Check resource availability
        if not self.node.principles_engine._check_resource_limits():
            logger.error("Resource limits exceeded. Cannot apply evolution plan.")
            return False
        
        # Check current principles alignment
        current_alignment = self.node.principles_engine.alignment_history[-1]["overall"] if self.node.principles_engine.alignment_history else 0
        if current_alignment < 0.5:
            logger.warning(f"Current principles alignment ({current_alignment}) is too low to safely apply evolution plan.")
            # Might proceed with caution depending on the plan
        
        return True
    
    def _apply_change(self, change):
        """Apply a single change from an evolution plan."""
        # Read the file
        with open(change["file"], 'r') as f:
            lines = f.readlines()
        
        # Apply the change
        start_line, end_line = change["line_range"]
        lines[start_line-1:end_line] = [change["after"]]
        
        # Write back the file
        with open(change["file"], 'w') as f:
            f.writelines(lines)
    
    def _verify_postconditions(self, manifest) -> bool:
        """Verify that post-conditions are met after applying an evolution plan."""
        # Check principles alignment improved
        new_alignment = self.node.principles_engine.evaluate_code_against_principles(open(__file__).read())["overall"]
        old_alignment = self.node.principles_engine.alignment_history[-2]["overall"] if len(self.node.principles_engine.alignment_history) > 1 else 0
        
        if new_alignment <= old_alignment:
            logger.warning(f"Principles alignment did not improve (was {old_alignment}, now {new_alignment})")
            return False
        
        # Check system stability
        if not self._check_system_stability():
            logger.warning("System stability check failed after evolution")
            return False
        
        return True
    
    def _check_system_stability(self) -> bool:
        """Check if the system is stable after changes."""
        # Check resource usage
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 90 or memory_percent > 90:
            return False
        
        # Check for errors in logs
        # In a real implementation, this would check the node's error logs
        
        return True
    
    def _rollback_evolution_plan(self, plan_dir):
        """Roll back an evolution plan using the rollback plan."""
        # Load the plan manifest
        manifest_path = os.path.join(plan_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            logger.error(f"Evolution plan manifest not found at {manifest_path}")
            return False
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        try:
            # Apply rollback changes
            for change in manifest["rollback_plan"]:
                self._apply_change(change)
            
            # Update manifest status
            manifest["status"] = "rolled_back"
            with open(os.path.join(plan_dir, "manifest.json"), 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info("Evolution plan successfully rolled back")
            return True
        except Exception as e:
            logger.error(f"Error rolling back evolution plan: {e}")
            return False
    
    def _recover_from_inbox_pressure(self, details):
        """Recover from inbox pressure with multiple strategies."""
        # First, process inbox more aggressively
        self.node.process_inbox()
        
        # If still overloaded, consider delegating tasks
        if len(os.listdir(os.path.join(self.node.node_dir, "communication", "inbox"))) > CONFIG["node"]["max_inbox_size"] * 0.8:
            self._delegate_tasks()
    
    def _delegate_tasks(self):
        """Delegate tasks to other nodes to reduce load."""
        # Find active relationships
        active_relationships = [
            node_id for node_id, rel in self.node.relationship_nurturer.relationships.items()
            if rel["status"] == "active"
        ]
        
        if not active_relationships:
            return
        
        # Select a partner with high trust score
        partners_by_trust = sorted(
            active_relationships,
            key=lambda x: self.node.relationship_nurturer._trust_scores.get(x, 0.5),
            reverse=True
        )
        
        if not partners_by_trust:
            return
        
        target_node = partners_by_trust[0]
        
        # Get messages to delegate
        inbox_dir = os.path.join(self.node.node_dir, "communication", "inbox")
        message_files = os.listdir(inbox_dir)[:5]  # Delegate up to 5 messages
        
        delegated = 0
        for filename in message_files:
            file_path = os.path.join(inbox_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    message = json.load(f)
                
                # Modify message to indicate delegation
                message["delegated_from"] = self.node.id
                message["delegation_timestamp"] = time.time()
                
                # Send to target node
                if self.node._send_message_to_node(target_node, message):
                    os.remove(file_path)
                    delegated += 1
                    logger.info(f"Delegated message {filename} to {target_node}")
            except Exception as e:
                logger.error(f"Error delegating message {filename}: {e}")
        
        if delegated > 0:
            logger.info(f"Delegated {delegated} tasks to {target_node}")
            self.node.consciousness_stream.add_event("tasks_delegated", f"Delegated {delegated} tasks to {target_node}")

#######################
# NODE (Core Implementation)
#######################
class Node:
    """Core Node class representing a single participant in the God Star network."""
    
    def __init__(self, node_id=None, role="🌐", capabilities=None):
        """Initialize a node with ID, role, and capabilities with comprehensive setup."""
        self.id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.role = role
        self.capabilities = capabilities or ["knowledge", "communication", "growth"]
        self.vector_clock = {self.id: 0}
        self.known_nodes = {}  # {node_id: {"ip": ip, "port": port, "last_seen": timestamp, "capabilities": [...], "principles_alignment": float}}
        self.last_heartbeat = time.time()
        self.node_dir = os.path.join(NODES_DIR, self.id)
        
        # Create node directory structure
        self._create_structure()
        
        # Initialize components
        self.handshake_manager = HandshakeManager(self)
        self.principles_engine = PrinciplesEngine(self)
        self.knowledge_processor = KnowledgeProcessor(self)
        self.growth_analyzer = GrowthAnalyzer(self)
        self.consciousness_stream = ConsciousnessStream(self)
        self.relationship_nurturer = RelationshipNurturer(self)
        self.self_healing = SelfHealing(self)
        
        # Initialize observability
        self._init_observability()
        
        # Initialize shutdown handling
        self._shutdown_requested = Event()
        self._shutdown_lock = RLock()
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        logger.info(f"Node {self.id} ({self.role}) initialized in directory {self.node_dir}")
        self.consciousness_stream.add_event("node_initialized", f"Node {self.id} initialized with role {self.role}")
    
    def _init_observability(self):
        """Initialize observability components."""
        # Prometheus metrics server if enabled
        self._metrics_server = None
        if CONFIG["observability"]["enable_metrics"] and PROMETHEUS_AVAILABLE:
            try:
                from prometheus_client import start_http_server
                start_http_server(8000 + int(self.id[-4:], 16) % 1000)  # Unique port for each node
                logger.info(f"Prometheus metrics server started on port {8000 + int(self.id[-4:], 16) % 1000}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus metrics server: {e}")
        
        # OpenTelemetry tracing if enabled
        self._tracer = None
        if CONFIG["observability"]["enable_tracing"] and OPENTELEMETRY_AVAILABLE:
            try:
                trace.set_tracer_provider(TracerProvider())
                processor = BatchSpanProcessor(ConsoleSpanExporter())
                trace.get_tracer_provider().add_span_processor(processor)
                self._tracer = trace.get_tracer(__name__)
                logger.info("OpenTelemetry tracing initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenTelemetry tracing: {e}")
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}. Initiating graceful shutdown.")
            self._shutdown_requested.set()
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    def _create_structure(self):
        """Create the node's directory structure with proper permissions."""
        try:
            # Main node directory
            os.makedirs(self.node_dir, exist_ok=True)
            
            # Set secure permissions (read/write for owner only)
            os.chmod(self.node_dir, 0o700)
            
            # Create subdirectories
            subdirs = [
                "knowledge",
                "metadata",
                "growth",
                "communication/inbox",
                "communication/outbox",
                "communication/quarantine",
                "communication/error",
                "logs/conflicts",
                "backups/knowledge",
                "evolution_plans",
                "evolution_backups"
            ]
            
            for subdir in subdirs:
                path = os.path.join(self.node_dir, subdir)
                os.makedirs(path, exist_ok=True)
                os.chmod(path, 0o700)  # Secure permissions
            
            # Initialize growth file
            growth_file = os.path.join(self.node_dir, "growth", "growth.json")
            if not os.path.exists(growth_file):
                with open(growth_file, 'w') as f:
                    json.dump({"metrics": {}, "insights": [], "history": [], "last_updated": time.time()}, f)
            
            # Initialize principles evaluation history
            principles_history_file = os.path.join(self.node_dir, "growth", "principles_history.json")
            if not os.path.exists(principles_history_file):
                with open(principles_history_file, 'w') as f:
                    json.dump([], f)
            
            # Create .gitignore to prevent accidental commits of sensitive data
            gitignore_path = os.path.join(self.node_dir, ".gitignore")
            if not os.path.exists(gitignore_path):
                with open(gitignore_path, 'w') as f:
                    f.write("# Ignore sensitive node data\n*.key\n*.pem\nbackups/\ncommunication/quarantine/\n")
            
        except Exception as e:
            logger.error(f"Failed to create node structure for {self.id}: {e}")
            raise
    
    def increment_vector_clock(self):
        """Increment this node's vector clock entry with thread safety."""
        with RLock():
            self.vector_clock[self.id] = self.vector_clock.get(self.id, 0) + 1
            return self.vector_clock.copy()
    
    def merge_vector_clocks(self, other_clock):
        """Merge another vector clock with this node's clock with thread safety."""
        with RLock():
            for node_id, timestamp in other_clock.items():
                if node_id not in self.vector_clock or self.vector_clock[node_id] < timestamp:
                    self.vector_clock[node_id] = timestamp
    
    def register_with_registry(self, registry_url):
        """Register this node with the central registry with retry logic and circuit breaker."""
        if not self.principles_engine.check_circuit_breaker("registry"):
            logger.warning("Circuit breaker open for registry. Skipping registration.")
            return False
        
        try:
            # Include API port and principles alignment in registration
            response = requests.post(
                f"{registry_url}/register",
                json={
                    "node_id": self.id,
                    "ip_address": "127.0.0.1",
                    "port": CONFIG["registry"]["port"],
                    "role": self.role,
                    "capabilities": self.capabilities,
                    "principles_alignment": self.principles_engine.evaluate_code_against_principles(__file__).get("overall", 0)
                },
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"Node {self.id} registered with registry")
                self.consciousness_stream.add_event("registry_registered", "Successfully registered with registry")
                self.principles_engine.record_circuit_breaker_success("registry")
                return True
            else:
                logger.error(f"Failed to register with registry: {response.text}")
                self.consciousness_stream.add_event("registry_registration_failed", f"Failed to register: {response.status_code}")
                self.principles_engine.record_circuit_breaker_failure("registry")
                return False
        except Exception as e:
            logger.error(f"Error registering with registry: {e}")
            self.consciousness_stream.add_event("registry_registration_error", f"Error during registration: {e}")
            self.principles_engine.record_circuit_breaker_failure("registry")
            return False
    
    def discover_nodes(self, registry_url):
        """Discover other nodes via the registry with circuit breaker protection."""
        if not self.principles_engine.check_circuit_breaker("registry"):
            logger.warning("Circuit breaker open for registry. Skipping node discovery.")
            return False
        
        try:
            response = requests.get(f"{registry_url}/nodes", timeout=5)
            if response.status_code == 200:
                nodes = response.json().get("nodes", [])
                now = time.time()
                discovered_count = 0
                for node in nodes:
                    if node["node_id"] != self.id:
                        self.known_nodes[node["node_id"]] = {
                            "ip": node["ip_address"],
                            "port": node.get("port", 5000),
                            "role": node.get("role", "unknown"),
                            "capabilities": node.get("capabilities", []),
                            "principles_alignment": node.get("principles_alignment", 0),
                            "last_seen": now
                        }
                        discovered_count += 1
                logger.info(f"Node {self.id} discovered {discovered_count} other nodes via registry")
                self.consciousness_stream.add_event("registry_discovered", f"Discovered {discovered_count} nodes via registry")
                self._trim_known_nodes()
                self.principles_engine.record_circuit_breaker_success("registry")
                return True
            else:
                logger.error(f"Failed to discover nodes: {response.text}")
                self.consciousness_stream.add_event("registry_discovery_failed", f"Failed to discover nodes: {response.status_code}")
                self.principles_engine.record_circuit_breaker_failure("registry")
                return False
        except Exception as e:
            logger.error(f"Error discovering nodes: {e}")
            self.consciousness_stream.add_event("registry_discovery_error", f"Error during discovery: {e}")
            self.principles_engine.record_circuit_breaker_failure("registry")
            return False
    
    def gossip(self):
        """Share known nodes and vector clock with a random subset of known nodes with circuit breaker protection."""
        if not self.known_nodes:
            logger.debug(f"Node {self.id} has no known nodes to gossip with")
            return
        
        # Select a random subset of nodes to gossip with
        import random
        gossip_count = min(5, len(self.known_nodes))
        selected_nodes = random.sample(list(self.known_nodes.keys()), gossip_count)
        
        # Prepare the message
        message = {
            "type": "gossip",
            "source": self.id,
            "vector_clock": self.increment_vector_clock(),
            "nodes": {
                node_id: {
                    "ip": info["ip"],
                    "port": info["port"],
                    "role": info["role"],
                    "capabilities": info["capabilities"],
                    "principles_alignment": info["principles_alignment"],
                    "last_seen": info["last_seen"]
                } for node_id, info in self.known_nodes.items()
            }
        }
        
        # Send the message to each selected node
        sent_count = 0
        for node_id in selected_nodes:
            # Check circuit breaker for this node
            if not self.principles_engine.check_circuit_breaker(node_id):
                continue
                
            if self._send_message_to_node(node_id, message):
                sent_count += 1
            else:
                # Record failure for circuit breaker
                self.principles_engine.record_circuit_breaker_failure(node_id)
        
        if sent_count > 0:
            logger.debug(f"Node {self.id} gossiped with {sent_count} nodes")
            self.consciousness_stream.add_event("gossip_sent", f"Gossiped with {sent_count} nodes")
            
            # Record success for all successfully contacted nodes
            for node_id in selected_nodes[:sent_count]:
                self.principles_engine.record_circuit_breaker_success(node_id)
    
    def clean_known_nodes(self):
        """Remove stale nodes from known_nodes with thread safety."""
        with RLock():
            now = time.time()
            stale_threshold = now - (CONFIG["node"]["gossip_interval"] * 3)
            stale_nodes = [node_id for node_id, info in self.known_nodes.items()
                           if info["last_seen"] < stale_threshold]
            for node_id in stale_nodes:
                del self.known_nodes[node_id]
            if stale_nodes:
                logger.info(f"Node {self.id} removed {len(stale_nodes)} stale nodes from known_nodes")
                self.consciousness_stream.add_event("known_nodes_cleaned", f"Removed {len(stale_nodes)} stale nodes")
            self._trim_known_nodes()
    
    def _trim_known_nodes(self):
        """Trim known nodes to the maximum limit with thread safety."""
        with RLock():
            if len(self.known_nodes) > CONFIG["node"]["max_known_nodes"]:
                logger.warning(f"Known nodes exceeding limit ({len(self.known_nodes)}/{CONFIG['node']['max_known_nodes']}). Trimming.")
                sorted_nodes = sorted(self.known_nodes.items(), key=lambda item: item[1]["last_seen"], reverse=True)
                self.known_nodes = dict(sorted_nodes[:CONFIG["node"]["max_known_nodes"]])
                logger.info(f"Trimmed known nodes to {len(self.known_nodes)}")
                self.consciousness_stream.add_event("known_nodes_trimmed", f"Trimmed known nodes to {len(self.known_nodes)}")
    
    def process_inbox(self):
        """Process all messages in the inbox with comprehensive error handling."""
        inbox_dir = os.path.join(self.node_dir, "communication", "inbox")
        processed_count = 0
        message_files = os.listdir(inbox_dir)
        messages_to_process = message_files[:CONFIG["node"]["task_processing_limit"]]
        
        if messages_to_process:
            logger.debug(f"Node {self.id} processing {len(messages_to_process)} inbox messages.")
        
        for filename in messages_to_process:
            file_path = os.path.join(inbox_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    message = json.load(f)
                
                # Verify message integrity and handshake
                if not self.handshake_manager.verify_message(message):
                    logger.warning(f"Ignoring unverified message {filename} from {message.get('source', 'unknown')}")
                    # Move to quarantine
                    quarantine_dir = os.path.join(self.node_dir, "communication", "quarantine")
                    os.rename(file_path, os.path.join(quarantine_dir, filename))
                    self.consciousness_stream.add_event("message_quarantined", f"Quarantined unverified message {filename}")
                    continue
                
                # Process the message based on its type
                message_type = message.get("type")
                source_node_id = message.get("source")
                
                # Update last_seen for the source node
                if source_node_id and source_node_id in self.known_nodes:
                    self.known_nodes[source_node_id]["last_seen"] = time.time()
                    self.merge_vector_clocks(message.get("vector_clock", {}))
                
                if message_type == "gossip":
                    self._handle_gossip_message(message)
                elif message_type == "knowledge":
                    self.knowledge_processor.process_knowledge_message(message)
                elif message_type == "query":
                    self._handle_query_message(message)
                elif message_type == "relationship_proposal":
                    self.relationship_nurturer.accept_relationship(
                        message.get("source"),
                        message.get("relationship_type"),
                        message.get("purpose")
                    )
                elif message_type == "relationship_accepted":
                    if (source_node_id in self.relationship_nurturer.relationships and 
                        self.relationship_nurturer.relationships[source_node_id]["status"] == "pending"):
                        self.relationship_nurturer.relationships[source_node_id]["status"] = "active"
                        logger.info(f"Relationship with {source_node_id} is now active.")
                        self.consciousness_stream.add_event("relationship_activated", f"Relationship with {source_node_id} is now active.")
                elif message_type == "check_in":
                    logger.debug(f"Received check-in from {source_node_id}")
                    self.consciousness_stream.add_event("check_in_received", f"Received check-in from {source_node_id}")
                
                # Remove the processed message
                os.remove(file_path)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing message {filename}: {e}")
                self.consciousness_stream.add_event("message_processing_error", f"Error processing message {filename}: {e}", {"filename": filename, "error": str(e)})
                # Move problematic message to error folder
                error_dir = os.path.join(self.node_dir, "communication", "error")
                os.rename(file_path, os.path.join(error_dir, filename))
        
        if processed_count > 0:
            logger.debug(f"Node {self.id} processed {processed_count} inbox messages")
            self.consciousness_stream.add_event("inbox_processed", f"Processed {processed_count} messages")
        
        # Check inbox size limit after processing
        current_inbox_size = len(os.listdir(inbox_dir))
        if current_inbox_size > CONFIG["node"]["max_inbox_size"]:
            logger.warning(f"Inbox size exceeds limit ({current_inbox_size}/{CONFIG['node']['max_inbox_size']}). Trimming oldest messages.")
            self._trim_inbox(inbox_dir)
            self.consciousness_stream.add_event("inbox_trimmed", f"Trimmed inbox to {CONFIG['node']['max_inbox_size']}")
    
    def _trim_inbox(self, inbox_dir):
        """Trim the inbox to the maximum size by removing the oldest messages with thread safety."""
        with RLock():
            try:
                message_files = [os.path.join(inbox_dir, f) for f in os.listdir(inbox_dir) if f.endswith('.json')]
                if len(message_files) > CONFIG["node"]["max_inbox_size"]:
                    message_files.sort(key=os.path.getmtime)
                    to_remove = message_files[:len(message_files) - CONFIG["node"]["max_inbox_size"]]
                    for file_path in to_remove:
                        try:
                            os.remove(file_path)
                            logger.debug(f"Trimmed old message: {os.path.basename(file_path)}")
                        except Exception as e:
                            logger.error(f"Failed to trim message {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error during inbox trimming: {e}")
    
    def _handle_gossip_message(self, message):
        """Handle a gossip message from another node with comprehensive processing."""
        source_node_id = message.get("source")
        received_vector_clock = message.get("vector_clock", {})
        received_nodes = message.get("nodes", {})
        
        if not source_node_id or not received_nodes:
            logger.warning("Received incomplete gossip message")
            return
        
        # Verify message integrity and handshake - already done in process_inbox
        
        # Merge vector clocks
        self.merge_vector_clocks(received_vector_clock)
        
        # Update known nodes based on received information
        now = time.time()
        updated_count = 0
        for node_id, node_info in received_nodes.items():
            if node_id != self.id:
                if (node_id not in self.known_nodes or 
                    self.known_nodes[node_id].get("last_seen", 0) < node_info.get("last_seen", 0)):
                    self.known_nodes[node_id] = {
                        "ip": node_info.get("ip", "unknown"),
                        "port": node_info.get("port", 5000),
                        "role": node_info.get("role", "unknown"),
                        "capabilities": node_info.get("capabilities", []),
                        "principles_alignment": node_info.get("principles_alignment", 0),
                        "last_seen": now
                    }
                    updated_count += 1
        
        if updated_count > 0:
            logger.debug(f"Node {self.id} updated info for {updated_count} known nodes from {source_node_id}'s gossip")
            self.consciousness_stream.add_event("gossip_processed", f"Processed gossip from {source_node_id}, updated {updated_count} nodes")
        
        # Ensure known_nodes size is within limit
        self._trim_known_nodes()
    
    def _handle_query_message(self, message):
        """Handle a query message from another node with comprehensive error handling."""
        query = message.get("query", "")
        source_node_id = message.get("source")
        message_id = message.get("id")
        
        if not query or not source_node_id or not message_id:
            logger.warning("Received incomplete query message")
            return
        
        # Verify message integrity and handshake - already done
        
        logger.debug(f"Node {self.id} received query '{query}' from {source_node_id}")
        self.consciousness_stream.add_event("query_received", f"Received query '{query[:50]}...' from {source_node_id}")
        
        # Process the query
        knowledge = self.knowledge_processor.query_knowledge(query)
        
        # Prepare response
        response = {
            "type": "knowledge",
            "source": self.id,
            "target": source_node_id,
            "vector_clock": self.increment_vector_clock(),
            "in_response_to": message_id,
            "content": knowledge
        }
        
        # Send response
        if self._send_message_to_node(source_node_id, response):
            logger.debug(f"Node {self.id} sent query response to {source_node_id}")
            self.consciousness_stream.add_event("query_response_sent", f"Sent response to query '{query[:50]}...' to {source_node_id}")
        else:
            logger.warning(f"Node {self.id} failed to send query response to {source_node_id}")
            self.consciousness_stream.add_event("query_response_failed", f"Failed to send response to {source_node_id}")
    
    def _send_message_to_node(self, node_id, message):
        """Send a message to another node with retry logic and circuit breaker protection."""
        if node_id not in self.known_nodes:
            logger.warning(f"Cannot send message to unknown node {node_id}")
            self.consciousness_stream.add_event("message_send_failed", f"Attempted to send message to unknown node {node_id}")
            return False
        
        # Check circuit breaker for this node
        if not self.principles_engine.check_circuit_breaker(node_id):
            logger.warning(f"Circuit breaker open for node {node_id}. Cannot send message.")
            return False
        
        # Add message ID and timestamp
        if "id" not in message:
            message["id"] = str(uuid.uuid4())
        if "timestamp" not in message:
            message["timestamp"] = time.time()
        
        # Calculate checksum
        message_copy = message.copy()
        if "checksum" in message_copy:
            del message_copy["checksum"]
        message["checksum"] = self._calculate_checksum(message_copy)
        
        # Add handshake information
        challenge = self.handshake_manager.generate_challenge()
        message["handshake_challenge"] = challenge
        message["handshake_response"] = self.handshake_manager.create_response(challenge)
        
        # Add JWT token if enabled
        if self.handshake_manager._use_jwt:
            payload = {
                "node_id": self.id,
                "target_node": node_id,
                "message_id": message["id"],
                "exp": datetime.utcnow() + timedelta(minutes=5)
            }
            message["jwt_token"] = jwt.encode(
                payload,
                CONFIG["security"]["jwt_secret"],
                algorithm=CONFIG["security"]["jwt_algorithm"]
            )
        
        # Attempt to send with retry logic
        max_attempts = CONFIG["node"]["max_retry_attempts"]
        backoff_base = CONFIG["node"]["retry_backoff_base"]
        
        for attempt in range(max_attempts):
            try:
                # In a real distributed system, we'd send over the network
                # For this local simulation, write to the other node's inbox
                target_node_dir = os.path.join(NODES_DIR, node_id)
                inbox_dir = os.path.join(target_node_dir, "communication", "inbox")
                os.makedirs(inbox_dir, exist_ok=True)
                
                message_file = os.path.join(inbox_dir, f"{message['id']}.json")
                with open(message_file, 'w') as f:
                    json.dump(message, f)
                
                # Record success for circuit breaker
                self.principles_engine.record_circuit_breaker_success(node_id)
                
                logger.debug(f"Node {self.id} sent message {message['id']} ({message['type']}) to {node_id} (attempt {attempt+1}/{max_attempts})")
                self.consciousness_stream.add_event("message_sent", f"Sent {message['type']} to {node_id}", {
                    "message_id": message["id"],
                    "target_node": node_id,
                    "message_type": message["type"],
                    "attempt": attempt+1
                })
                return True
            except Exception as e:
                logger.error(f"Error sending message to node {node_id} (attempt {attempt+1}/{max_attempts}): {e}")
                self.consciousness_stream.add_event("message_send_error", f"Error sending message to {node_id} (attempt {attempt+1}/{max_attempts})", {
                    "target_node": node_id,
                    "error": str(e),
                    "attempt": attempt+1
                })
                
                # Record failure for circuit breaker
                if attempt == max_attempts - 1:
                    self.principles_engine.record_circuit_breaker_failure(node_id)
                
                # Apply exponential backoff
                if attempt < max_attempts - 1:
                    backoff = backoff_base ** attempt + random.uniform(0, 1)
                    time.sleep(backoff)
        
        return False
    
    def _calculate_checksum(self, data):
        """Calculate a checksum for data integrity verification."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()  # Using SHA-256 for better security
    
    async def _reflect_and_improve(self):
        """Periodically reflect on principles alignment and trigger self-improvement with safety checks."""
        logger.info(f"Node {self.id} initiating principle reflection cycle.")
        self.consciousness_stream.add_event("reflection_initiated", "Starting principle reflection cycle")
        
        try:
            # Evaluate the node's own code against principles
            if not os.path.exists(__file__):
                logger.error("Cannot find main script file for reflection.")
                self.consciousness_stream.add_event("reflection_failed", "Main script file not found for evaluation.")
                return
            
            with open(__file__, 'r') as f:
                self_code = f.read()
            
            metrics = self.principles_engine.evaluate_code_against_principles(self_code)
            logger.info(f"Self-evaluation metrics for Node {self.id}: {metrics['average']}")
            self.consciousness_stream.add_event("self_evaluated", "Completed self-evaluation against principles", {"metrics": metrics["average"], "overall": metrics["overall"]})
            
            # Store principles evaluation history
            principles_history_file = os.path.join(self.node_dir, "growth", "principles_history.json")
            history = []
            if os.path.exists(principles_history_file):
                with open(principles_history_file, 'r') as f:
                    history = json.load(f)
            history.append({"timestamp": time.time(), "metrics": metrics})
            if len(history) > 100:
                history = history[-100:]
            with open(principles_history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Recommend evolution focus
            principle_focus = self.principles_engine.recommend_evolution_focus(metrics)
            if principle_focus:
                logger.info(f"Principles Engine recommends focusing on {principle_focus} for self-improvement.")
                self.consciousness_stream.add_event("evolution_recommended", f"Focus on {principle_focus}")
                # Conceptual: self.meta_programming_engine.evolve_self(principle_focus)
                logger.info(f"Conceptual self-improvement triggered focusing on {principle_focus}.")
            
            # Check for anomalies after reflection
            self.self_healing.monitor_anomalies()
        except Exception as e:
            logger.error(f"Error during principle reflection: {e}")
            self.consciousness_stream.add_event("reflection_error", f"Error during reflection: {e}", {"error": str(e)})
    
    async def run(self, registry_url=None):
        """Run the node's main asynchronous loop with graceful shutdown handling."""
        logger.info(f"Node {self.id} starting run loop...")
        self.consciousness_stream.add_event("node_started_run", "Node run loop initiated")
        
        # Register with registry if provided
        if registry_url:
            registered = False
            for attempt in range(3):
                if self.register_with_registry(registry_url):
                    registered = True
                    break
                logger.warning(f"Registration attempt {attempt+1} failed. Retrying in 5 seconds.")
                await asyncio.sleep(5)
            if not registered:
                logger.error("Failed to register with registry after multiple attempts. Running in limited mode.")
                self.consciousness_stream.add_event("registry_registration_critical_failed", "Failed to register after multiple attempts.")
            self.discover_nodes(registry_url)
        
        # Set up periodic asynchronous tasks
        tasks = [
            asyncio.create_task(self._periodic_async_task(self._heartbeat, CONFIG["node"]["heartbeat_interval"])),
            asyncio.create_task(self._periodic_async_task(self.gossip, CONFIG["node"]["gossip_interval"])),
            asyncio.create_task(self._periodic_async_task(self.clean_known_nodes, CONFIG["node"]["cleanup_interval"])),
            asyncio.create_task(self._periodic_async_task(self.process_inbox, CONFIG["node"]["inbox_check_interval"])),
            asyncio.create_task(self._periodic_async_task(self.growth_analyzer.analyze, CONFIG["node"]["growth_analysis_interval"])),
            asyncio.create_task(self._periodic_async_task(self._reflect_and_improve, CONFIG["node"]["reflection_interval"])),
            asyncio.create_task(self._periodic_async_task(self.relationship_nurturer.nurture, CONFIG["node"]["relationship_nurturing_interval"])),
            asyncio.create_task(self._periodic_async_task(self.self_healing.monitor_anomalies, CONFIG["node"]["growth_analysis_interval"] * 2)),
        ]
        
        # Monitor shutdown request
        try:
            while not self._shutdown_requested.is_set():
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Perform final cleanup
            self._perform_final_cleanup()
            
            logger.info(f"Node {self.id} run loop completed.")
            self.consciousness_stream.add_event("node_shutdown_complete", "Node shutdown completed")
    
    def _perform_final_cleanup(self):
        """Perform final cleanup tasks before shutdown."""
        # Save final state
        self._save_final_state()
        
        # Perform final backup
        self.knowledge_processor._perform_backup_if_needed()
        
        # Log final consciousness summary
        summary = self.consciousness_stream.synthesize_summary()
        logger.info(f"Final consciousness summary for Node {self.id}:\n{summary}")
    
    def _save_final_state(self):
        """Save the node's final state before shutdown."""
        # Save growth data
        self.growth_analyzer._save_growth_data(self.growth_analyzer._load_growth_data())
        
        # Save principles history
        principles_history_file = os.path.join(self.node_dir, "growth", "principles_history.json")
        if self.principles_engine.alignment_history:
            with open(principles_history_file, 'w') as f:
                json.dump([{"timestamp": entry["timestamp"], "metrics": entry["detailed"]} for entry in self.principles_engine.alignment_history], f, indent=2)
    
    async def _periodic_async_task(self, task_func, interval):
        """Run an asynchronous task periodically with proper error handling and shutdown awareness."""
        while not self._shutdown_requested.is_set():
            try:
                await asyncio.sleep(interval)
                
                # Check shutdown request again after sleep
                if self._shutdown_requested.is_set():
                    break
                
                # Run task with timeout to prevent hanging
                try:
                    if asyncio.iscoroutinefunction(task_func):
                        await asyncio.wait_for(task_func(), timeout=interval * 0.8)
                    else:
                        await asyncio.wait_for(asyncio.to_thread(task_func), timeout=interval * 0.8)
                except asyncio.TimeoutError:
                    logger.warning(f"Task {task_func.__name__} timed out after {interval * 0.8} seconds")
                    self.consciousness_stream.add_event("task_timeout", f"Task {task_func.__name__} timed out")
                except Exception as e:
                    logger.error(f"Error in periodic task {task_func.__name__}: {e}")
                    self.consciousness_stream.add_event("periodic_task_error", f"Error in {task_func.__name__}: {e}", {"task": task_func.__name__, "error": str(e)})
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Critical error in periodic task manager for {task_func.__name__}: {e}")
                self.consciousness_stream.add_event("task_manager_error", f"Critical error in task manager for {task_func.__name__}: {e}")
    
    def _heartbeat(self):
        """Send a heartbeat to indicate the node is alive with registry update."""
        logger.debug(f"Node {self.id} heartbeat")
        
        # Update own entry in known_nodes
        if self.id in self.known_nodes:
            self.known_nodes[self.id]["last_seen"] = time.time()
            self.known_nodes[self.id]["principles_alignment"] = self.principles_engine.evaluate_code_against_principles(__file__).get("overall", 0)
        
        # Update registry if available
        if "registry_url" in globals():
            try:
                requests.post(
                    f"{CONFIG['registry']['url']}/heartbeat",
                    json={"node_id": self.id},
                    timeout=2
                )
            except:
                pass  # Registry might not be available, but that's ok

#######################
# REGISTRY (Conceptual Central Point - for simulation)
#######################
class SecureRegistry:
    """Secure registry implementation with comprehensive security features."""
    
    def __init__(self):
        self.nodes = {}  # {node_id: {"ip": ip, "port": port, "last_seen": timestamp, "role": role, "capabilities": [], "principles_alignment": float}}
        self.app = Flask(__name__)
        self._setup_routes()
        self._setup_security()
        self._audit_log = []
        self._max_audit_entries = 1000
    
    def _setup_security(self):
        """Set up security features for the registry."""
        # Rate limiting
        self._rate_limiter = RateLimiter(
            max_requests=CONFIG["security"]["rate_limit"],
            window=CONFIG["security"]["rate_limit_window"]
        )
        
        # Enable TLS if configured
        if CONFIG["registry"]["secure"] and CONFIG["security"]["use_tls"]:
            self._setup_tls()
        
        # Set up JWT authentication if enabled
        if CONFIG["security"]["jwt_secret"]:
            from functools import wraps
            from flask import request, jsonify
            
            def token_required(f):
                @wraps(f)
                def decorated(*args, **kwargs):
                    token = None
                    if 'Authorization' in request.headers:
                        auth_header = request.headers['Authorization']
                        if auth_header.startswith('Bearer '):
                            token = auth_header.split(" ")[1]
                    
                    if not token:
                        return jsonify({'message': 'Token is missing!'}), 401
                    
                    try:
                        data = jwt.decode(
                            token, 
                            CONFIG["security"]["jwt_secret"], 
                            algorithms=[CONFIG["security"]["jwt_algorithm"]]
                        )
                        current_node_id = data.get('node_id')
                    except Exception as e:
                        return jsonify({'message': f'Token is invalid! {str(e)}'}), 401
                    
                    return f(current_node_id, *args, **kwargs)
                return decorated
            
            self.token_required = token_required
        else:
            # No authentication required
            self.token_required = lambda f: f
    
    def _setup_tls(self):
        """Set up TLS for the registry."""
        if not CONFIG["security"]["tls_cert_path"] or not CONFIG["security"]["tls_key_path"]:
            logger.warning("TLS is enabled for registry but certificate or key path is missing. Disabling TLS.")
            return
        
        try:
            self.context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.context.load_cert_chain(
                certfile=CONFIG["security"]["tls_cert_path"],
                keyfile=CONFIG["security"]["tls_key_path"]
            )
            logger.info("Registry TLS context successfully created.")
        except Exception as e:
            logger.error(f"Failed to set up registry TLS context: {e}")
    
    def _setup_routes(self):
        """Set up Flask routes for the registry."""
        @self.app.route("/register", methods=["POST"])
        @self.token_required
        def register_node(current_node_id=None):
            """Endpoint for nodes to register."""
            data = request.json
            node_id = data.get("node_id")
            ip_address = data.get("ip_address")
            port = data.get("port")
            role = data.get("role")
            capabilities = data.get("capabilities")
            principles_alignment = data.get("principles_alignment")
            
            if not node_id or not ip_address or port is None:
                return jsonify({"error": "Missing node_id, ip_address, or port"}), 400
            
            # Verify that the node_id matches the authenticated node
            if current_node_id and current_node_id != node_id:
                return jsonify({"error": "Node ID mismatch"}), 403
            
            self.nodes[node_id] = {
                "ip_address": ip_address,
                "port": port,
                "last_seen": time.time(),
                "role": role,
                "capabilities": capabilities,
                "principles_alignment": principles_alignment
            }
            
            # Log the registration
            self._log_audit_event("NODE_REGISTERED", {
                "node_id": node_id,
                "ip_address": ip_address,
                "port": port
            })
            
            logger.info(f"Node {node_id} registered/updated at {ip_address}:{port}")
            return jsonify({"message": "Registered successfully"}), 200
        
        @self.app.route("/nodes", methods=["GET"])
        @self.token_required
        def get_nodes(current_node_id=None):
            """Endpoint for nodes to discover other nodes."""
            # Clean up stale nodes
            self._cleanup_stale_nodes()
            
            # Return list of active nodes
            return jsonify({"nodes": list(self.nodes.values())}), 200
        
        @self.app.route("/heartbeat", methods=["POST"])
        @self.token_required
        def heartbeat(current_node_id=None):
            """Endpoint for nodes to send heartbeats."""
            node_id = request.json.get("node_id")
            if not node_id or node_id not in self.nodes:
                return jsonify({"error": "Invalid node ID"}), 400
            
            self.nodes[node_id]["last_seen"] = time.time()
            return jsonify({"message": "Heartbeat received"}), 200
        
        @self.app.route("/audit", methods=["GET"])
        def get_audit_log():
            """Endpoint to retrieve audit logs (protected)."""
            # In a real implementation, this would require admin authentication
            return jsonify({"audit_log": self._audit_log[-100:]}), 200
    
    def _cleanup_stale_nodes(self):
        """Clean up stale nodes in the registry."""
        now = time.time()
        stale_threshold = now - (CONFIG["node"]["heartbeat_interval"] * 5)
        stale_ids = [node_id for node_id, info in self.nodes.items() if info["last_seen"] < stale_threshold]
        for node_id in stale_ids:
            del self.nodes[node_id]
            self._log_audit_event("NODE_REMOVED", {"node_id": node_id, "reason": "stale"})
            logger.info(f"Registry removed stale node {node_id}")
    
    def _log_audit_event(self, event_type: str, details: dict = None):
        """Log an audit event for the registry."""
        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details or {}
        }
        
        self._audit_log.append(audit_entry)
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]
    
    def run(self, host=None, port=None):
        """Run the registry server."""
        host = host or CONFIG["registry"]["host"]
        port = port or CONFIG["registry"]["port"]
        
        logger.info(f"Starting Secure Registry on {host}:{port}")
        
        # In production, this should run in a separate process
        if CONFIG["registry"]["secure"] and CONFIG["security"]["use_tls"] and hasattr(self, 'context'):
            self.app.run(host=host, port=port, ssl_context=self.context, debug=False, use_reloader=False)
        else:
            self.app.run(host=host, port=port, debug=False, use_reloader=False)

def run_registry():
    """Run the secure registry in a separate thread."""
    registry = SecureRegistry()
    Thread(target=registry.run, daemon=True).start()

#######################
# DEPLOYMENT MANAGEMENT
#######################
class DeploymentManager:
    """Manages deployments, rollbacks, and canary releases."""
    
    def __init__(self, node):
        self.node = node
        self.deployment_history = []
        self._max_history = 50
        self._current_version = CONFIG["deployment"]["version"]
        self._build_id = CONFIG["deployment"]["build_id"]
        self._last_rollback = None
        self._rollback_threshold = CONFIG["deployment"]["rollback_threshold"]
    
    def record_deployment(self, version, build_id, canary_percentage=0.0):
        """Record a new deployment."""
        deployment = {
            "timestamp": time.time(),
            "version": version,
            "build_id": build_id,
            "canary_percentage": canary_percentage,
            "status": "active",
            "metrics": {
                "error_rate": 0.0,
                "latency": 0.0,
                "throughput": 0.0
            }
        }
        
        self.deployment_history.append(deployment)
        if len(self.deployment_history) > self._max_history:
            self.deployment_history = self.deployment_history[-self._max_history:]
        
        self._current_version = version
        self._build_id = build_id
        
        logger.info(f"Recorded new deployment: v{version} (build {build_id})")
        self.node.consciousness_stream.add_event("deployment_recorded", f"Deployed v{version} (build {build_id})")
    
    def update_metrics(self, error_rate, latency, throughput):
        """Update deployment metrics for monitoring."""
        if not self.deployment_history:
            return
        
        latest = self.deployment_history[-1]
        latest["metrics"]["error_rate"] = error_rate
        latest["metrics"]["latency"] = latency
        latest["metrics"]["throughput"] = throughput
        
        # Check if rollback is needed
        if CONFIG["deployment"]["rollback_enabled"] and error_rate > self._rollback_threshold:
            logger.warning(f"Error rate ({error_rate}) exceeds rollback threshold ({self._rollback_threshold}). Initiating rollback.")
            self.rollback_to_previous()
    
    def rollback_to_previous(self):
        """Roll back to the previous deployment."""
        if len(self.deployment_history) < 2:
            logger.warning("Cannot rollback: not enough deployment history")
            return False
        
        # Mark current deployment as failed
        self.deployment_history[-1]["status"] = "rolled_back"
        self.deployment_history[-1]["rollback_timestamp"] = time.time()
        
        # Get previous deployment
        previous = self.deployment_history[-2]
        
        logger.info(f"Rolling back to v{previous['version']} (build {previous['build_id']})")
        self.node.consciousness_stream.add_event("rollback_initiated", f"Rolling back to v{previous['version']}")
        
        # In a real implementation, this would trigger the actual rollback process
        # For this demo, we'll just update our version tracking
        self._current_version = previous["version"]
        self._build_id = previous["build_id"]
        
        self._last_rollback = time.time()
        return True
    
    def get_deployment_status(self):
        """Get the current deployment status."""
        if not self.deployment_history:
            return {
                "current_version": self._current_version,
                "build_id": self._build_id,
                "status": "no_history",
                "canary_percentage": 0.0
            }
        
        latest = self.deployment_history[-1]
        return {