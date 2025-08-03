######################## BIO-INSPIRED IMMUNE SYSTEM #######################
class NodeImmuneSystem:
    """Implements a biological immune system analog for node network protection and healing"""
    def __init__(self, node):
        self.node = node
        self.memory_cells = {}  # Stores patterns of past threats
        self.antibody_production_rate = 0.7  # How quickly nodes respond to threats
        self.tolerance_threshold = 0.3  # How much anomaly a node tolerates before responding
        self.autoimmune_risk = 0.05  # Risk of mistakenly attacking healthy nodes
        
        # Immune cell types with different functions
        self.immune_cells = {
            "macrophages": [],  # Identify and consume threats
            "b_cells": [],      # Remember past threats (memory)
            "t_cells": [],      # Coordinate immune response
            "dendritic": []     # Present "threat antigens" to other nodes
        }
        
        # Immune response history for learning
        self.response_history = []
        self.max_history = 1000
    
    def monitor_health(self):
        """Continuously monitor node health using biological analogs"""
        # Calculate "inflammatory markers" based on system metrics
        inflammatory_markers = self._calculate_inflammatory_markers()
        
        # Check for "fever" (high resource usage)
        fever_level = self._detect_fever()
        
        # Check for "infection vectors" (suspicious network activity)
        infection_risk = self._assess_infection_risk()
        
        # Overall immune activation level
        activation_level = (inflammatory_markers * 0.4 + 
                           fever_level * 0.3 + 
                           infection_risk * 0.3)
        
        # Store in consciousness stream
        if activation_level > self.tolerance_threshold * 0.5:
            self.node.consciousness_stream.add_event(
                "immune_activation", 
                f"Immune system activated at level {activation_level:.2f}",
                {"inflammatory": inflammatory_markers, "fever": fever_level, "infection_risk": infection_risk}
            )
            
            # Trigger appropriate immune response
            self._trigger_immune_response(activation_level)
            
        return activation_level
    
    def _calculate_inflammatory_markers(self):
        """Calculate markers analogous to biological inflammation indicators"""
        # CRP (C-Reactive Protein) analog - measures system instability
        crp = min(1.0, self.node.growth_analyzer.get_system_instability() * 2)
        
        # IL-6 analog - measures communication anomalies
        il6 = min(1.0, self._calculate_communication_anomalies() * 1.5)
        
        # TNF-alpha analog - measures security threats
        tnf = min(1.0, self._calculate_security_threat_level() * 1.2)
        
        # Weighted average as overall inflammation marker
        return (crp * 0.4 + il6 * 0.3 + tnf * 0.3)
    
    def _detect_fever(self):
        """Detect "fever" - abnormal resource consumption patterns"""
        # Normal baseline is 0.3-0.4 for CPU/memory
        current_load = (self.node.system_monitor.get_cpu_percent() * 0.6 + 
                       self.node.system_monitor.get_memory_percent() * 0.4) / 100
        
        # Fever is proportional to how far above normal baseline
        baseline = 0.35
        if current_load <= baseline:
            return 0.0
        
        # Non-linear response - small increases are normal, large spikes are concerning
        fever = min(1.0, (current_load - baseline) * 3.0)
        return fever ** 2  # Square for non-linear response
    
    def _calculate_communication_anomalies(self):
        """Detect abnormal communication patterns like a biological immune system"""
        # Get recent communication patterns
        recent_events = self.node.consciousness_stream.get_stream(limit=100)
        
        # Look for patterns that resemble infection vectors
        anomaly_score = 0.0
        message_bursts = 0
        unusual_partners = 0
        
        # Check for message bursts (like viral replication)
        message_counts = defaultdict(int)
        for event in recent_events:
            if event["type"] == "message_received":
                message_counts[event["timestamp"] // 60] += 1  # Count per minute
        
        # Detect unusual bursts
        if message_counts:
            avg_rate = sum(message_counts.values()) / len(message_counts)
            for count in message_counts.values():
                if count > avg_rate * 5:  # 5x normal rate
                    message_bursts += min(1.0, (count - avg_rate) / avg_rate * 0.2)
        
        # Check for communication with unknown or low-trust nodes
        for event in recent_events:
            if event["type"] == "message_received" and "target_node_id" in event:
                trust = self.node.relationship_nurturer.get_trust_score(event["target_node_id"])
                if trust < 0.3:  # Low trust connection
                    unusual_partners += 0.1
        
        anomaly_score = min(1.0, (message_bursts * 0.7 + unusual_partners * 0.3))
        return anomaly_score
    
    def _calculate_security_threat_level(self):
        """Assess security threats using immune system analogs"""
        # Check recent security events
        security_events = [
            e for e in self.node.consciousness_stream.get_stream(limit=50)
            if e["type"].startswith("security_") or e["type"].endswith("_failed")
        ]
        
        threat_level = 0.0
        for event in security_events:
            # Weight different event types
            if "handshake" in event["type"]:
                threat_level += 0.1
            elif "authentication" in event["type"]:
                threat_level += 0.2
            elif "breach" in event["type"]:
                threat_level += 0.5
            elif "critical" in event["type"]:
                threat_level += 0.8
        
        return min(1.0, threat_level)
    
    def _trigger_immune_response(self, threat_level):
        """Trigger appropriate immune response based on threat level"""
        if threat_level < self.tolerance_threshold:
            return  # No response needed
        
        # Determine response type based on threat level
        if threat_level < 0.5:
            # Mild response - just monitor more closely
            self._activate_macrophages(threat_level)
        elif threat_level < 0.7:
            # Moderate response - isolate potential threats
            self._activate_t_cells(threat_level)
        else:
            # Severe response - quarantine and call for help
            self._activate_full_response(threat_level)
    
    def _activate_macrophages(self, threat_level):
        """Activate "macrophages" to consume minor threats"""
        # In node terms: increase monitoring and clean up minor issues
        logger.info(f"Activating macrophages for threat level {threat_level:.2f}")
        
        # Clean up temporary files
        self.node.knowledge_processor.cleanup_temp_files()
        
        # Reset minor counters
        self.node.rate_limiter.reset_counters()
        
        # Record in immune history
        self._record_immune_response("macrophage_activation", threat_level, "minor_threat_cleanup")
    
    def _activate_t_cells(self, threat_level):
        """Activate "T-cells" to coordinate response to moderate threats"""
        logger.info(f"Activating T-cells for threat level {threat_level:.2f}")
        
        # Isolate suspicious nodes
        suspicious_nodes = self._identify_suspicious_nodes()
        for node_id in suspicious_nodes:
            self.node.relationship_nurturer.isolate_node(node_id, reason="immune_response")
            logger.warning(f"Isolated node {node_id} as part of immune response")
        
        # Request assistance from trusted nodes
        if len(suspicious_nodes) > 0:
            self._request_immune_assistance(suspicious_nodes)
        
        # Record in immune history
        self._record_immune_response("t_cell_activation", threat_level, 
                                   f"isolated {len(suspicious_nodes)} nodes")
    
    def _activate_full_response(self, threat_level):
        """Activate full immune response for severe threats"""
        logger.critical(f"ACTIVATING FULL IMMUNE RESPONSE for threat level {threat_level:.2f}")
        
        # Quarantine compromised components
        self._quarantine_compromised_components()
        
        # Activate emergency communication with trusted nodes
        self._broadcast_emergency()
        
        # Initiate self-healing protocols
        self.node.self_healing.assess_and_heal("severe_system_threat", 
                                             {"immune_threat_level": threat_level})
        
        # Record in immune history
        self._record_immune_response("full_response", threat_level, "system_quarantine_initiated")
    
    def _identify_suspicious_nodes(self):
        """Identify nodes showing suspicious behavior using immune memory"""
        suspicious = []
        
        # Check recent communication for anomalies
        for node_id, node_info in self.node.known_nodes.items():
            # Get trust score
            trust = self.node.relationship_nurturer.get_trust_score(node_id)
            
            # Check immune memory for past issues
            if node_id in self.memory_cells and self.memory_cells[node_id]["severity"] > 0.5:
                suspicious.append(node_id)
                continue
                
            # Check for recent anomalies
            recent_events = [
                e for e in self.node.consciousness_stream.get_stream(limit=50)
                if e.get("target_node_id") == node_id and "security" in e.get("type", "")
            ]
            
            if len(recent_events) > 3 or trust < 0.3:
                suspicious.append(node_id)
        
        return suspicious
    
    def _quarantine_compromised_components(self):
        """Quarantine potentially compromised system components"""
        # In node terms: isolate compromised services
        logger.warning("Quarantining potentially compromised components")
        
        # Stop non-essential services
        self.node.stop_non_essential_services()
        
        # Activate read-only mode for knowledge base
        self.node.knowledge_processor.activate_read_only_mode()
        
        # Record in immune history
        self._record_immune_response("quarantine", 1.0, "system_components_quarantined")
    
    def _broadcast_emergency(self):
        """Broadcast emergency signal to trusted nodes"""
        trusted_nodes = [
            node_id for node_id in self.node.known_nodes
            if self.node.relationship_nurturer.get_trust_score(node_id) > 0.7
        ]
        
        if trusted_nodes:
            emergency_msg = {
                "type": "EMERGENCY_IMMUNE_RESPONSE",
                "node_id": self.node.id,
                "threat_level": 1.0,
                "timestamp": time.time(),
                "request_help": True
            }
            
            # Send to random subset of trusted nodes to avoid overload
            target_nodes = random.sample(trusted_nodes, min(3, len(trusted_nodes)))
            for node_id in target_nodes:
                self.node._send_message_to_node(node_id, emergency_msg)
    
    def _record_immune_response(self, response_type, threat_level, details):
        """Record immune response in history for learning"""
        response = {
            "timestamp": time.time(),
            "response_type": response_type,
            "threat_level": threat_level,
            "details": details,
            "node_state": self._capture_node_state()
        }
        
        self.response_history.append(response)
        if len(self.response_history) > self.max_history:
            self.response_history.pop(0)
        
        # Update memory cells for future reference
        if threat_level > self.tolerance_threshold:
            self._update_memory_cells(response)
    
    def _capture_node_state(self):
        """Capture current node state for immune memory"""
        return {
            "cpu": self.node.system_monitor.get_cpu_percent(),
            "memory": self.node.system_monitor.get_memory_percent(),
            "active_relationships": len(self.node.known_nodes),
            "principles_alignment": self.node.principles_engine.get_alignment_score(),
            "recent_events": [e["type"] for e in self.node.consciousness_stream.get_stream(limit=10)]
        }
    
    def _update_memory_cells(self, immune_response):
        """Update immune memory with new threat patterns"""
        # Extract patterns from the response
        threat_pattern = self._extract_threat_pattern(immune_response)
        
        # Store in memory cells with decay rate
        if threat_pattern["signature"] not in self.memory_cells:
            self.memory_cells[threat_pattern["signature"]] = {
                "pattern": threat_pattern,
                "first_occurrence": time.time(),
                "last_occurrence": time.time(),
                "severity": threat_pattern["threat_level"],
                "response_effectiveness": 0.0,  # Will be updated after response
                "decay_rate": 0.95  # Memory fades over time
            }
        else:
            # Update existing memory cell
            cell = self.memory_cells[threat_pattern["signature"]]
            cell["last_occurrence"] = time.time()
            cell["severity"] = max(cell["severity"], threat_pattern["threat_level"])
            # Update effectiveness after we know outcome