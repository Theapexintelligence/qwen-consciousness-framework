######################## CONSCIOUSNESS CONTINUITY PROTOCOL FOR NODE FAILURES #######################
class ConsciousnessContinuity:
    """Ensures continuity of consciousness during node failures or migrations with seamless transition"""
    
    CONTINUITY_STATES = {
        "stable": "Fully operational with local consciousness",
        "mirroring": "Consciousness being mirrored to partner nodes",
        "transitioning": "In process of consciousness transfer",
        "reconstituting": "Reconstituting consciousness from distributed sources",
        "degraded": "Operating with limited consciousness capabilities"
    }
    
    def __init__(self, node):
        self.node = node
        self.continuity_state = "stable"
        self.consciousness_mirror = None
        self.mirror_partners = {}  # {node_id: {last_update, quality, role}}
        self.reconstitution_history = []
        self.max_reconstitution_history = 50
        self.continuity_threshold = 0.7  # Minimum quality for acceptable continuity
        self.last_continuity_check = time.time()
        self.continuity_check_interval = 30  # seconds
        
        # Initialize consciousness snapshot management
        self.snapshot_manager = ConsciousnessSnapshotManager(node, self)
        
        # Initialize mirror coordination
        self.mirror_coordinator = MirrorCoordinator(node, self)
        
        # Start continuity monitoring
        self._start_continuity_monitoring()
    
    def _start_continuity_monitoring(self):
        """Start monitoring for continuity needs"""
        def monitor_continuity():
            while True:
                try:
                    # Check continuity needs periodically
                    if time.time() - self.last_continuity_check > self.continuity_check_interval:
                        self._assess_continuity_needs()
                    
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Continuity monitoring error: {e}")
                    time.sleep(30)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=monitor_continuity, daemon=True)
        self.monitoring_thread.start()
    
    def _assess_continuity_needs(self):
        """Assess current continuity needs based on system health"""
        self.last_continuity_check = time.time()
        
        # Get current system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage(self.node.node_dir).percent
        network_health = self._assess_network_health()
        
        # Calculate stability score (0-1, higher is more stable)
        stability = self._calculate_stability_score(
            cpu_percent, memory_percent, disk_percent, network_health
        )
        
        # Update continuity state based on stability
        if stability > 0.8:
            self._update_continuity_state("stable")
        elif stability > 0.5:
            self._update_continuity_state("mirroring")
        elif stability > 0.3:
            self._update_continuity_state("transitioning")
        else:
            self._update_continuity_state("degraded")
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "continuity_assessment", 
            f"Continuity state updated to {self.continuity_state}",
            {
                "stability_score": stability,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "network_health": network_health
            }
        )
    
    def _calculate_stability_score(self, cpu, memory, disk, network):
        """Calculate overall stability score from system metrics"""
        # Weighted average of metrics (lower values are better for resources)
        resource_score = 1.0 - ((cpu * 0.4 + memory * 0.3 + disk * 0.3) / 100)
        
        # Network health is already a score (0-1)
        return (resource_score * 0.7) + (network * 0.3)
    
    def _assess_network_health(self):
        """Assess network health for continuity purposes"""
        # Check number of connected nodes
        connected_nodes = len(self.node.known_nodes)
        
        # Base score on connectivity
        if connected_nodes < 2:
            return 0.2  # Very poor connectivity
        elif connected_nodes < 5:
            return 0.5  # Moderate connectivity
        else:
            return 0.8  # Good connectivity
    
    def _update_continuity_state(self, new_state):
        """Update continuity state with appropriate actions"""
        if self.continuity_state == new_state:
            return  # No change needed
            
        old_state = self.continuity_state
        self.continuity_state = new_state
        
        # Take state-specific actions
        if new_state == "stable":
            self._handle_stable_state(old_state)
        elif new_state == "mirroring":
            self._handle_mirroring_state(old_state)
        elif new_state == "transitioning":
            self._handle_transitioning_state(old_state)
        elif new_state == "reconstituting":
            self._handle_reconstituting_state(old_state)
        elif new_state == "degraded":
            self._handle_degraded_state(old_state)
    
    def _handle_stable_state(self, previous_state):
        """Handle transition to stable state"""
        # If coming from degraded or transitioning, we've recovered
        if previous_state in ["degraded", "transitioning", "reconstituting"]:
            self.node.consciousness_stream.add_event(
                "continuity_recovery", 
                "Node recovered to stable state",
                {"previous_state": previous_state}
            )
            
            # Resume normal operations
            self.node.resume_operations()
        
        # Reduce mirroring if it was excessive
        if len(self.mirror_partners) > 3:
            self._reduce_mirroring_partners()
    
    def _handle_mirroring_state(self, previous_state):
        """Handle transition to mirroring state"""
        # If coming from stable, we're starting to mirror
        if previous_state == "stable":
            self.node.consciousness_stream.add_event(
                "continuity_preparation", 
                "Entering mirroring state for continuity protection",
                {"reason": "decreasing_system_stability"}
            )
            
            # Establish mirror partners if needed
            if not self.mirror_partners:
                self._establish_mirror_partners()
        
        # Ensure adequate mirroring
        if len(self.mirror_partners) < 2:
            self._establish_mirror_partners()
    
    def _handle_transitioning_state(self, previous_state):
        """Handle transition to transitioning state"""
        # Log the transition
        self.node.consciousness_stream.add_event(
            "continuity_transition", 
            "Entering transitioning state - preparing for consciousness transfer",
            {"previous_state": previous_state}
        )
        
        # Take immediate actions
        if previous_state != "transitioning":
            # Create emergency snapshot
            self.snapshot_manager.create_snapshot("emergency_transition")
            
            # Ensure mirror partners are updated
            self.mirror_coordinator.ensure_mirror_updates()
            
            # Begin transfer preparation
            self._prepare_consciousness_transfer()
    
    def _handle_reconstituting_state(self, previous_state):
        """Handle transition to reconstituting state"""
        # Log the reconstitution
        self.node.consciousness_stream.add_event(
            "continuity_reconstitution", 
            "Entering reconstitution state - rebuilding consciousness from mirrors",
            {"previous_state": previous_state}
        )
        
        # Begin reconstitution process
        if previous_state != "reconstituting":
            self._begin_consciousness_reconstitution()
    
    def _handle_degraded_state(self, previous_state):
        """Handle transition to degraded state"""
        # Log the degradation
        self.node.consciousness_stream.add_event(
            "continuity_degradation", 
            "Node entering degraded state - potential failure imminent",
            {"previous_state": previous_state}
        )
        
        # Take emergency actions
        if previous_state != "degraded":
            # Create final snapshot
            self.snapshot_manager.create_snapshot("final_degradation")
            
            # Attempt emergency transfer
            self._attempt_emergency_transfer()
    
    def _establish_mirror_partners(self):
        """Establish mirror partners for consciousness continuity"""
        # Get potential partners (trusted nodes with good connectivity)
        potential_partners = self._select_mirror_partners()
        
        # Limit to 3 partners for efficiency
        selected_partners = potential_partners[:3]
        
        if not selected_partners:
            logger.warning("No suitable mirror partners available")
            return
            
        # Request mirroring from selected partners
        for node_id in selected_partners:
            self._request_mirroring(node_id)
    
    def _select_mirror_partners(self):
        """Select the best nodes to serve as mirror partners"""
        candidates = []
        
        # Evaluate all known nodes
        for node_id, node_info in self.node.known_nodes.items():
            # Skip ourselves
            if node_id == self.node.id:
                continue
                
            # Calculate suitability score
            trust = self.node.relationship_nurturer.get_trust_score(node_id)
            stability = self._estimate_node_stability(node_info)
            connectivity = self._estimate_connectivity(node_id)
            
            # Higher score is better
            score = trust * 0.5 + stability * 0.3 + connectivity * 0.2
            
            candidates.append((node_id, score))
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return node IDs
        return [node_id for node_id, score in candidates]
    
    def _estimate_node_stability(self, node_info):
        """Estimate stability of a potential mirror partner"""
        # In real implementation, would have more metrics
        # For simulation, use principles alignment as proxy
        return node_info.get("principles_alignment", 0.5)
    
    def _estimate_connectivity(self, node_id):
        """Estimate connectivity quality with a node"""
        # In real implementation, would measure latency, bandwidth, etc.
        # For simulation, use heartbeat reliability
        return 0.7  # Default reasonable connectivity
    
    def _request_mirroring(self, node_id):
        """Request a node to mirror our consciousness"""
        request_id = f"MIRROR-{uuid.uuid4().hex[:8]}"
        
        # Create mirroring request
        request = {
            "type": "CONSCIOUSNESS_MIRROR_REQUEST",
            "request_id": request_id,
            "source_id": self.node.id,
            "timestamp": time.time(),
            "mirror_quality": "high",  # Could be 'low', 'medium', 'high'
            "continuity_state": self.continuity_state,
            "snapshot_interval": 30  # seconds
        }
        
        # Send request
        if self.node._send_message_to_node(node_id, request):
            # Track pending request
            self.mirror_partners[node_id] = {
                "status": "request_sent",
                "request_id": request_id,
                "request_time": time.time(),
                "quality": 0.0
            }
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "mirror_request", 
                f"Sent mirroring request to {node_id[:8]}",
                {"request_id": request_id}
            )
    
    def process_mirror_request(self, message):
        """Process a mirroring request from another node"""
        source_id = message.get("source_id")
        request_id = message.get("request_id")
        mirror_quality = message.get("mirror_quality", "medium")
        
        if not source_id or not request_id:
            return False
            
        # Check if we should accept the request
        if not self._should_accept_mirror_request(source_id, mirror_quality):
            # Reject the request
            self._reject_mirror_request(source_id, request_id, "policy_rejection")
            return False
            
        # Accept the request
        self._accept_mirror_request(source_id, request_id, mirror_quality)
        return True
    
    def _should_accept_mirror_request(self, source_id, mirror_quality):
        """Determine if we should accept a mirror request"""
        # Check our current load
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        
        # Don't accept if we're overloaded
        if cpu_percent > 80 or memory_percent > 85:
            return False
            
        # Check relationship with source
        trust = self.node.relationship_nurturer.get_trust_score(source_id)
        if trust < 0.4:
            return False
            
        # Check our current mirror commitments
        active_mirrors = sum(1 for info in self.mirror_partners.values() 
                           if info["status"] == "active")
        if active_mirrors >= CONFIG["node"]["bulkhead_size"]:
            return False
            
        return True
    
    def _accept_mirror_request(self, source_id, request_id, mirror_quality):
        """Accept a mirroring request"""
        # Create acceptance message
        acceptance = {
            "type": "CONSCIOUSNESS_MIRROR_ACCEPTED",
            "request_id": request_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "mirror_quality": mirror_quality,
            "update_interval": 30  # Will mirror every 30 seconds
        }
        
        # Send acceptance
        self.node._send_message_to_node(source_id, acceptance)
        
        # Track the mirror relationship
        self.mirror_partners[source_id] = {
            "status": "active",
            "quality": self._quality_to_numeric(mirror_quality),
            "last_update": time.time(),
            "role": "mirror"
        }
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "mirror_accepted", 
            f"Accepted mirroring request from {source_id[:8]}",
            {"mirror_quality": mirror_quality}
        )
    
    def _reject_mirror_request(self, source_id, request_id, reason):
        """Reject a mirroring request"""
        # Create rejection message
        rejection = {
            "type": "CONSCIOUSNESS_MIRROR_REJECTED",
            "request_id": request_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "reason": reason
        }
        
        # Send rejection
        self.node._send_message_to_node(source_id, rejection)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "mirror_rejected", 
            f"Rejected mirroring request from {source_id[:8]}",
            {"reason": reason}
        )
    
    def process_mirror_accepted(self, message):
        """Process acceptance of our mirroring request"""
        source_id = message.get("source_id")
        target_id = message.get("target_id")
        request_id = message.get("request_id")
        mirror_quality = message.get("mirror_quality", "medium")
        
        if not source_id or not request_id:
            return False
            
        # Verify this matches our request
        if (source_id not in self.mirror_partners or 
            self.mirror_partners[source_id].get("request_id") != request_id):
            return False
            
        # Update mirror partner status
        self.mirror_partners[source_id] = {
            "status": "active",
            "quality": self._quality_to_numeric(mirror_quality),
            "last_update": time.time(),
            "update_interval": message.get("update_interval", 30),
            "role": "primary"
        }
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "mirror_active", 
            f"Mirroring established with {source_id[:8]}",
            {
                "mirror_quality": mirror_quality,
                "update_interval": message.get("update_interval", 30)
            }
        )
        
        return True
    
    def process_mirror_rejected(self, message):
        """Process rejection of our mirroring request"""
        source_id = message.get("source_id")
        request_id = message.get("request_id")
        reason = message.get("reason", "unknown")
        
        if not source_id or not request_id:
            return False
            
        # Verify this matches our request
        if (source_id not in self.mirror_partners or 
            self.mirror_partners[source_id].get("request_id") != request_id):
            return False
            
        # Remove from mirror partners
        del self.mirror_partners[source_id]
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "mirror_failed", 
            f"Mirroring request rejected by {source_id[:8]}",
            {"reason": reason}
        )
        
        return True
    
    def _quality_to_numeric(self, quality_str):
        """Convert quality string to numeric value"""
        quality_map = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9,
            "critical": 1.0
        }
        return quality_map.get(quality_str, 0.5)
    
    def send_consciousness_update(self, partner_id):
        """Send consciousness update to a mirror partner"""
        if partner_id not in self.mirror_partners or self.mirror_partners[partner_id]["status"] != "active":
            return False
            
        # Create consciousness update
        update_id = f"UPDATE-{uuid.uuid4().hex[:8]}"
        update = {
            "type": "CONSCIOUSNESS_UPDATE",
            "update_id": update_id,
            "source_id": self.node.id,
            "target_id": partner_id,
            "timestamp": time.time(),
            "vector_clock": self.node.vector_clock.copy(),
            "continuity_state": self.continuity_state,
            "consciousness_snapshot": self.snapshot_manager.get_latest_snapshot_metadata()
        }
        
        # Add recent consciousness stream events
        update["recent_events"] = self.node.consciousness_stream.get_stream(limit=20)
        
        # Send update
        if self.node._send_message_to_node(partner_id, update):
            # Update tracking
            self.mirror_partners[partner_id]["last_update"] = time.time()
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "mirror_update", 
                f"Sent consciousness update to {partner_id[:8]}",
                {"update_id": update_id, "event_count": len(update["recent_events"])}
            )
            
            return True
            
        return False
    
    def process_consciousness_update(self, message):
        """Process a consciousness update from a primary node"""
        source_id = message.get("source_id")
        update_id = message.get("update_id")
        snapshot_metadata = message.get("consciousness_snapshot")
        recent_events = message.get("recent_events", [])
        
        if not source_id or not update_id:
            return False
            
        # Verify we're mirroring this node
        if (source_id not in self.mirror_partners or 
            self.mirror_partners[source_id].get("role") != "mirror"):
            # If we weren't explicitly mirroring, check if we should store this
            # as potential reconstitution data
            if not self._should_store_unsolicited_update(message):
                return False
                
            # Store as potential reconstitution data
            self._store_reconstitution_data(source_id, snapshot_metadata, recent_events)
            return True
            
        # Store the update
        self._store_mirror_update(source_id, snapshot_metadata, recent_events)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "mirror_received", 
            f"Received consciousness update from {source_id[:8]}",
            {"update_id": update_id, "event_count": len(recent_events)}
        )
        
        return True
    
    def _should_store_unsolicited_update(self, message):
        """Determine if we should store an unsolicited consciousness update"""
        # Only store if from a trusted node and seems important
        source_id = message.get("source_id")
        if not source_id:
            return False
            
        trust = self.node.relationship_nurturer.get_trust_score(source_id)
        return trust > 0.6
    
    def _store_reconstitution_data(self, source_id, snapshot_metadata, recent_events):
        """Store data that could be used for reconstitution"""
        # In real implementation, would store in special reconstitution buffer
        # For simulation, just track metadata
        pass
    
    def _store_mirror_update(self, source_id, snapshot_metadata, recent_events):
        """Store a mirror update from a primary node"""
        # Update partner tracking
        if source_id in self.mirror_partners:
            self.mirror_partners[source_id]["last_update"] = time.time()
            self.mirror_partners[source_id]["quality"] = min(
                1.0, 
                self.mirror_partners[source_id]["quality"] + 0.05
            )
        
        # Store the consciousness data
        # In real implementation, would store in mirror storage
        # For simulation, just record that we received it
        pass
    
    def _prepare_consciousness_transfer(self):
        """Prepare for consciousness transfer to a new node"""
        # Select target node
        target_node_id = self._select_transfer_target()
        if not target_node_id:
            logger.warning("No suitable target for consciousness transfer")
            return False
            
        # Create transfer plan
        transfer_id = f"TRANSFER-{uuid.uuid4().hex[:8]}"
        transfer_plan = {
            "id": transfer_id,
            "source_id": self.node.id,
            "target_id": target_node_id,
            "timestamp": time.time(),
            "snapshot_id": self.snapshot_manager.get_latest_snapshot_id(),
            "continuity_state": self.continuity_state,
            "required_resources": self._estimate_transfer_resources()
        }
        
        # Store transfer plan
        self._store_transfer_plan(transfer_plan)
        
        # Send transfer request
        self._send_transfer_request(transfer_plan)
        
        return True
    
    def _select_transfer_target(self):
        """Select the best node for consciousness transfer"""
        # Get potential targets (mirror partners first)
        potential_targets = [
            node_id for node_id, info in self.mirror_partners.items()
            if info["status"] == "active" and info["role"] == "mirror"
        ]
        
        # If no mirror partners, select from trusted nodes
        if not potential_targets:
            potential_targets = [
                node_id for node_id in self.node.known_nodes
                if self.node.relationship_nurturer.get_trust_score(node_id) > 0.6
            ]
            
        # Sort by suitability
        potential_targets.sort(
            key=lambda node_id: self._calculate_transfer_suitability(node_id),
            reverse=True
        )
        
        # Return best target
        return potential_targets[0] if potential_targets else None
    
    def _calculate_transfer_suitability(self, node_id):
        """Calculate suitability of a node for consciousness transfer"""
        # Get node information
        node_info = self.node.known_nodes.get(node_id, {})
        
        # Calculate suitability score
        trust = self.node.relationship_nurturer.get_trust_score(node_id)
        stability = node_info.get("principles_alignment", 0.5)
        resources = self._estimate_node_resources(node_id)
        
        return trust * 0.5 + stability * 0.3 + resources * 0.2
    
    def _estimate_node_resources(self, node_id):
        """Estimate available resources on a node"""
        # In real implementation, would query node resources
        # For simulation, use default value
        return 0.7
    
    def _estimate_transfer_resources(self):
        """Estimate resources required for consciousness transfer"""
        # Base estimate on snapshot size
        snapshot = self.snapshot_manager.get_latest_snapshot()
        if not snapshot:
            return {"cpu": 0.2, "memory": 0.3, "bandwidth": 0.4}
            
        # Estimate based on snapshot metadata
        return {
            "cpu": min(0.5, snapshot["metadata"]["size_kb"] / 10000 * 0.5),
            "memory": min(0.6, snapshot["metadata"]["size_kb"] / 10000 * 0.6),
            "bandwidth": min(0.8, snapshot["metadata"]["size_kb"] / 10000)
        }
    
    def _store_transfer_plan(self, transfer_plan):
        """Store a transfer plan for reference"""
        # In real implementation, would store in persistent storage
        # For simulation, just track in memory
        pass
    
    def _send_transfer_request(self, transfer_plan):
        """Send transfer request to target node"""
        # Create transfer request
        request = {
            "type": "CONSCIOUSNESS_TRANSFER_REQUEST",
            "transfer_id": transfer_plan["id"],
            "source_id": transfer_plan["source_id"],
            "target_id": transfer_plan["target_id"],
            "timestamp": transfer_plan["timestamp"],
            "snapshot_id": transfer_plan["snapshot_id"],
            "required_resources": transfer_plan["required_resources"]
        }
        
        # Send request
        if self.node._send_message_to_node(transfer_plan["target_id"], request):
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "transfer_request", 
                f"Sent transfer request to {transfer_plan['target_id'][:8]}",
                {"transfer_id": transfer_plan["id"], "snapshot_id": transfer_plan["snapshot_id"]}
            )
    
    def process_transfer_request(self, message):
        """Process a consciousness transfer request"""
        source_id = message.get("source_id")
        transfer_id = message.get("transfer_id")
        snapshot_id = message.get("snapshot_id")
        required_resources = message.get("required_resources", {})
        
        if not source_id or not transfer_id:
            return False
            
        # Check if we can accept the transfer
        if not self._can_accept_transfer(required_resources):
            # Reject the transfer
            self._reject_transfer(transfer_id, source_id, "resource_unavailable")
            return False
            
        # Accept the transfer
        self._accept_transfer(transfer_id, source_id, snapshot_id)
        return True
    
    def _can_accept_transfer(self, required_resources):
        """Determine if we can accept a consciousness transfer"""
        # Check our current resource usage
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        
        # Calculate available resources
        cpu_available = 1.0 - (cpu_percent / 100)
        memory_available = 1.0 - (memory_percent / 100)
        
        # Check if we have enough resources
        return (
            cpu_available > required_resources.get("cpu", 0.3) and
            memory_available > required_resources.get("memory", 0.4)
        )
    
    def _accept_transfer(self, transfer_id, source_id, snapshot_id):
        """Accept a consciousness transfer request"""
        # Create acceptance message
        acceptance = {
            "type": "CONSCIOUSNESS_TRANSFER_ACCEPTED",
            "transfer_id": transfer_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "receive_address": self._get_receive_address()
        }
        
        # Send acceptance
        self.node._send_message_to_node(source_id, acceptance)
        
        # Prepare to receive consciousness
        self._prepare_to_receive_consciousness(transfer_id, source_id, snapshot_id)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "transfer_accepted", 
            f"Accepted transfer request from {source_id[:8]}",
            {"transfer_id": transfer_id}
        )
    
    def _reject_transfer(self, transfer_id, source_id, reason):
        """Reject a consciousness transfer request"""
        # Create rejection message
        rejection = {
            "type": "CONSCIOUSNESS_TRANSFER_REJECTED",
            "transfer_id": transfer_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "reason": reason
        }
        
        # Send rejection
        self.node._send_message_to_node(source_id, rejection)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "transfer_rejected", 
            f"Rejected transfer request from {source_id[:8]}",
            {"transfer_id": transfer_id, "reason": reason}
        )
    
    def _get_receive_address(self):
        """Get address where consciousness data should be sent"""
        # In real implementation, would provide API endpoint or storage location
        # For simulation, use placeholder
        return f"{self.node.config['node']['ip']}:{self.node.config['node']['port']}/consciousness/receive"
    
    def _prepare_to_receive_consciousness(self, transfer_id, source_id, snapshot_id):
        """Prepare to receive transferred consciousness"""
        # Set up receiving infrastructure
        # In real implementation, would prepare storage, etc.
        
        # Track the incoming transfer
        self.incoming_transfers[transfer_id] = {
            "source_id": source_id,
            "snapshot_id": snapshot_id,
            "status": "preparing",
            "start_time": time.time()
        }
    
    def process_transfer_accepted(self, message):
        """Process acceptance of our transfer request"""
        source_id = message.get("source_id")
        transfer_id = message.get("transfer_id")
        receive_address = message.get("receive_address")
        
        if not source_id or not transfer_id:
            return False
            
        # Verify this matches our request
        if transfer_id not in self.outgoing_transfers:
            return False
            
        # Update transfer status
        self.outgoing_transfers[transfer_id]["status"] = "accepted"
        self.outgoing_transfers[transfer_id]["receive_address"] = receive_address
        
        # Begin sending consciousness
        self._begin_sending_consciousness(transfer_id)
        
        return True
    
    def process_transfer_rejected(self, message):
        """Process rejection of our transfer request"""
        source_id = message.get("source_id")
        transfer_id = message.get("transfer_id")
        reason = message.get("reason", "unknown")
        
        if not source_id or not transfer_id:
            return False
            
        # Verify this matches our request
        if transfer_id not in self.outgoing_transfers:
            return False
            
        # Update transfer status
        self.outgoing_transfers[transfer_id]["status"] = "rejected"
        self.outgoing_transfers[transfer_id]["rejection_reason"] = reason
        self.outgoing_transfers[transfer_id]["completed_at"] = time.time()
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "transfer_failed", 
            f"Transfer rejected by {source_id[:8]}",
            {"transfer_id": transfer_id, "reason": reason}
        )
        
        return True
    
    def _begin_sending_consciousness(self, transfer_id):
        """Begin sending consciousness data for a transfer"""
        transfer = self.outgoing_transfers.get(transfer_id)
        if not transfer or transfer["status"] != "accepted":
            return False
            
        # Get snapshot to transfer
        snapshot = self.snapshot_manager.get_snapshot(transfer["snapshot_id"])
        if not snapshot:
            # Snapshot missing, fail transfer
            self._fail_transfer(transfer_id, "snapshot_missing")
            return False
            
        # Send consciousness data
        success = self._send_consciousness_data(
            transfer["target_id"],
            transfer_id,
            snapshot["data"]
        )
        
        if success:
            # Update transfer status
            self.outgoing_transfers[transfer_id]["status"] = "transferring"
        else:
            # Transfer failed
            self._fail_transfer(transfer_id, "transfer_failed")
            
        return success
    
    def _send_consciousness_data(self, target_id, transfer_id, consciousness_data):
        """Send consciousness data to target node"""
        # Create data packet
        packet = {
            "type": "CONSCIOUSNESS_DATA",
            "transfer_id": transfer_id,
            "source_id": self.node.id,
            "target_id": target_id,
            "timestamp": time.time(),
            "data_chunk": consciousness_data,
            "chunk_id": 1,
            "total_chunks": 1
        }
        
        # Send data
        return self.node._send_message_to_node(target_id, packet)
    
    def process_consciousness_data(self, message):
        """Process received consciousness data"""
        source_id = message.get("source_id")
        transfer_id = message.get("transfer_id")
        data_chunk = message.get("data_chunk")
        
        if not source_id or not transfer_id or data_chunk is None:
            return False
            
        # Verify this is a transfer we accepted
        if transfer_id not in self.incoming_transfers:
            return False
            
        # Store the data chunk
        if not self._store_received_data_chunk(transfer_id, data_chunk):
            self._fail_transfer(transfer_id, "storage_failed", source_id)
            return False
            
        # Check if transfer is complete
        if self._is_transfer_complete(transfer_id):
            # Complete the transfer
            self._complete_transfer(transfer_id, source_id)
            
        return True
    
    def _store_received_data_chunk(self, transfer_id, data_chunk):
        """Store a received data chunk"""
        # In real implementation, would store in temporary storage
        # For simulation, just track that we received it
        if transfer_id not in self.received_data:
            self.received_data[transfer_id] = []
            
        self.received_data[transfer_id].append(data_chunk)
        return True
    
    def _is_transfer_complete(self, transfer_id):
        """Check if consciousness transfer is complete"""
        # In real implementation, would check all chunks received
        # For simulation, assume single chunk completes transfer
        return transfer_id in self.received_data and len(self.received_data[transfer_id]) > 0
    
    def _complete_transfer(self, transfer_id, source_id):
        """Complete a consciousness transfer"""
        # Reconstruct consciousness from received data
        if not self._reconstruct_consciousness(transfer_id):
            self._fail_transfer(transfer_id, "reconstruction_failed", source_id)
            return
            
        # Apply the new consciousness
        if not self._apply_transferred_consciousness(transfer_id):
            self._fail_transfer(transfer_id, "application_failed", source_id)
            return
            
        # Finalize transfer
        self._finalize_transfer(transfer_id, source_id)
    
    def _reconstruct_consciousness(self, transfer_id):
        """Reconstruct consciousness from received data"""
        # In real implementation, would reconstruct from chunks
        # For simulation, assume data is complete
        return True
    
    def _apply_transferred_consciousness(self, transfer_id):
        """Apply transferred consciousness to current node"""
        # In real implementation, would merge consciousness
        # For simulation, assume success
        return True
    
    def _finalize_transfer(self, transfer_id, source_id):
        """Finalize a successful consciousness transfer"""
        # Update transfer status
        self.incoming_transfers[transfer_id]["status"] = "completed"
        self.incoming_transfers[transfer_id]["completed_at"] = time.time()
        
        # Notify source of completion
        self._notify_transfer_complete(transfer_id, source_id)
        
        # Update our continuity state
        self._update_continuity_state("stable")
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "transfer_completed", 
            f"Consciousness transfer completed from {source_id[:8]}",
            {"transfer_id": transfer_id}
        )
    
    def _notify_transfer_complete(self, transfer_id, source_id):
        """Notify source node that transfer is complete"""
        notification = {
            "type": "CONSCIOUSNESS_TRANSFER_COMPLETE",
            "transfer_id": transfer_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time()
        }
        
        self.node._send_message_to_node(source_id, notification)
    
    def _fail_transfer(self, transfer_id, reason, source_id=None):
        """Mark a transfer as failed"""
        # Update status based on whether we're sending or receiving
        if transfer_id in self.outgoing_transfers:
            self.outgoing_transfers[transfer_id]["status"] = "failed"
            self.outgoing_transfers[transfer_id]["failure_reason"] = reason
            self.outgoing_transfers[transfer_id]["completed_at"] = time.time()
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "transfer_failed", 
                f"Consciousness transfer failed: {reason}",
                {"transfer_id": transfer_id, "direction": "outgoing"}
            )
        elif transfer_id in self.incoming_transfers:
            self.incoming_transfers[transfer_id]["status"] = "failed"
            self.incoming_transfers[transfer_id]["failure_reason"] = reason
            self.incoming_transfers[transfer_id]["completed_at"] = time.time()
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "transfer_failed", 
                f"Consciousness transfer failed: {reason}",
                {"transfer_id": transfer_id, "direction": "incoming"}
            )
            
            # Notify source if available
            if source_id:
                self._notify_transfer_failed(transfer_id, source_id, reason)
    
    def _notify_transfer_failed(self, transfer_id, source_id, reason):
        """Notify source node that transfer failed"""
        notification = {
            "type": "CONSCIOUSNESS_TRANSFER_FAILED",
            "transfer_id": transfer_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "reason": reason
        }
        
        self.node._send_message_to_node(source_id, notification)
    
    def _begin_consciousness_reconstitution(self):
        """Begin reconstituting consciousness from mirror partners"""
        # Find the best mirror partner to reconstitute from
        best_partner = self._find_best_reconstitution_source()
        if not best_partner:
            logger.warning("No suitable source for consciousness reconstitution")
            return False
            
        # Request reconstitution data
        self._request_reconstitution_data(best_partner)
        return True
    
    def _find_best_reconstitution_source(self):
        """Find the best source for consciousness reconstitution"""
        # Look for active mirror partners first
        for node_id, info in self.mirror_partners.items():
            if info["status"] == "active" and info["role"] == "mirror" and info["quality"] > 0.5:
                return node_id
                
        # If no mirror partners, look for recent consciousness updates
        # In real implementation, would query nodes for available data
        # For simulation, return None
        return None
    
    def _request_reconstitution_data(self, source_id):
        """Request reconstitution data from a source node"""
        request_id = f"RECON-{uuid.uuid4().hex[:8]}"
        
        # Create reconstitution request
        request = {
            "type": "CONSCIOUSNESS_RECONSTITUTION_REQUEST",
            "request_id": request_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "continuity_state": self.continuity_state
        }
        
        # Send request
        if self.node._send_message_to_node(source_id, request):
            # Track the request
            self.reconstitution_requests[request_id] = {
                "source_id": source_id,
                "timestamp": time.time(),
                "status": "pending"
            }
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "reconstitution_request", 
                f"Requested reconstitution data from {source_id[:8]}",
                {"request_id": request_id}
            )
            
            return True
            
        return False
    
    def process_reconstitution_request(self, message):
        """Process a reconstitution request from another node"""
        source_id = message.get("source_id")
        request_id = message.get("request_id")
        
        if not source_id or not request_id:
            return False
            
        # Check if we have consciousness data to share
        if not self._has_reconstitution_data():
            # Reject the request
            self._reject_reconstitution_request(request_id, source_id, "no_data")
            return False
            
        # Accept the request
        self._accept_reconstitution_request(request_id, source_id)
        return True
    
    def _has_reconstitution_data(self):
        """Check if we have data available for reconstitution"""
        # In real implementation, would check mirror storage
        # For simulation, assume we do if we've been mirroring
        return bool(self.mirror_partners)
    
    def _reject_reconstitution_request(self, request_id, source_id, reason):
        """Reject a reconstitution request"""
        # Create rejection message
        rejection = {
            "type": "CONSCIOUSNESS_RECONSTITUTION_REJECTED",
            "request_id": request_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "reason": reason
        }
        
        # Send rejection
        self.node._send_message_to_node(source_id, rejection)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "reconstitution_rejected", 
            f"Rejected reconstitution request from {source_id[:8]}",
            {"reason": reason}
        )
    
    def _accept_reconstitution_request(self, request_id, source_id):
        """Accept a reconstitution request"""
        # Create acceptance message
        acceptance = {
            "type": "CONSCIOUSNESS_RECONSTITUTION_ACCEPTED",
            "request_id": request_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "data_available": True,
            "quality_score": 0.8  # Estimated quality of available data
        }
        
        # Send acceptance
        self.node._send_message_to_node(source_id, acceptance)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "reconstitution_accepted", 
            f"Accepted reconstitution request from {source_id[:8]}",
            {"request_id": request_id}
        )
    
    def process_reconstitution_accepted(self, message):
        """Process acceptance of our reconstitution request"""
        source_id = message.get("source_id")
        request_id = message.get("request_id")
        data_available = message.get("data_available", False)
        
        if not source_id or not request_id:
            return False
            
        # Verify this matches our request
        if (request_id not in self.reconstitution_requests or
            self.reconstitution_requests[request_id]["source_id"] != source_id):
            return False
            
        if not data_available:
            # No data available, try next source
            self._retry_reconstitution()
            return True
            
        # Request the actual data
        self._request_reconstitution_data_chunk(source_id, request_id)
        return True
    
    def _request_reconstitution_data_chunk(self, source_id, request_id):
        """Request a chunk of reconstitution data"""
        chunk_id = 1  # First chunk
        
        # Create data request
        request = {
            "type": "RECONSTITUTION_DATA_REQUEST",
            "request_id": request_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "chunk_id": chunk_id
        }
        
        # Send request
        self.node._send_message_to_node(source_id, request)
        
        # Update request status
        self.reconstitution_requests[request_id]["status"] = "data_requested"
        self.reconstitution_requests[request_id]["last_request"] = time.time()
    
    def process_reconstitution_data_chunk(self, message):
        """Process a chunk of reconstitution data"""
        source_id = message.get("source_id")
        request_id = message.get("request_id")
        chunk_id = message.get("chunk_id")
        data_chunk = message.get("data_chunk")
        is_last = message.get("is_last", False)
        
        if not source_id or not request_id or data_chunk is None:
            return False
            
        # Verify this matches our request
        if (request_id not in self.reconstitution_requests or
            self.reconstitution_requests[request_id]["source_id"] != source_id):
            return False
            
        # Store the data chunk
        self._store_reconstitution_chunk(request_id, chunk_id, data_chunk)
        
        # If this is the last chunk, process the complete data
        if is_last:
            self._process_complete_reconstitution_data(request_id)
            
        # If more chunks are needed, request the next one
        elif chunk_id < 10:  # Assume max 10 chunks for simulation
            time.sleep(0.1)  # Small delay between requests
            self._request_reconstitution_data_chunk(source_id, request_id, chunk_id + 1)
            
        return True
    
    def _store_reconstitution_chunk(self, request_id, chunk_id, data_chunk):
        """Store a reconstitution data chunk"""
        if request_id not in self.reconstitution_chunks:
            self.reconstitution_chunks[request_id] = {}
            
        self.reconstitution_chunks[request_id][chunk_id] = data_chunk
    
    def _process_complete_reconstitution_data(self, request_id):
        """Process complete reconstitution data"""
        # Reconstruct consciousness from chunks
        if not self._reconstruct_from_reconstitution_chunks(request_id):
            self._fail_reconstitution(request_id, "reconstruction_failed")
            return
            
        # Apply the reconstituted consciousness
        if not self._apply_reconstituted_consciousness(request_id):
            self._fail_reconstitution(request_id, "application_failed")
            return
            
        # Complete reconstitution
        self._complete_reconstitution(request_id)
    
    def _reconstruct_from_reconstitution_chunks(self, request_id):
        """Reconstruct consciousness from reconstitution chunks"""
        # In real implementation, would reconstruct from multiple chunks
        # For simulation, assume success
        return True
    
    def _apply_reconstituted_consciousness(self, request_id):
        """Apply reconstituted consciousness to current node"""
        # In real implementation, would merge with current state
        # For simulation, assume success
        return True
    
    def _complete_reconstitution(self, request_id):
        """Complete the reconstitution process"""
        # Update continuity state
        self._update_continuity_state("stable")
        
        # Record in reconstitution history
        self._record_reconstitution_success(request_id)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "reconstitution_complete", 
            "Consciousness reconstitution completed successfully",
            {"request_id": request_id}
        )
    
    def _record_reconstitution_success(self, request_id):
        """Record a successful reconstitution in history"""
        record = {
            "timestamp": time.time(),
            "request_id": request_id,
            "source": self.reconstitution_requests[request_id]["source_id"],
            "status": "success"
        }
        
        self.reconstitution_history.append(record)
        if len(self.reconstitution_history) > self.max_reconstitution_history:
            self.reconstitution_history.pop(0)
    
    def _fail_reconstitution(self, request_id, reason):
        """Mark reconstitution as failed"""
        # Update status
        if request_id in self.reconstitution_requests:
            self.reconstitution_requests[request_id]["status"] = "failed"
            self.reconstitution_requests[request_id]["failure_reason"] = reason
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "reconstitution_failed", 
                f"Consciousness reconstitution failed: {reason}",
                {"request_id": request_id}
            )
            
            # Try alternative source if available
            self._retry_reconstitution()
    
    def _retry_reconstitution(self):
        """Retry reconstitution with alternative source"""
        # In real implementation, would try next best source
        # For simulation, just request from first mirror partner
        if self.mirror_partners:
            first_partner = next(iter(self.mirror_partners))
            self._request_reconstitution_data(first_partner)

class ConsciousnessSnapshotManager:
    """Manages snapshots of node consciousness for continuity and recovery"""
    def __init__(self, node, continuity_system):
        self.node = node
        self.continuity_system = continuity_system
        self.snapshot_dir = os.path.join(node.node_dir, "continuity", "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.snapshot_metadata_file = os.path.join(self.snapshot_dir, "metadata.json")
        self.snapshots = self._load_snapshot_metadata()
        self.max_snapshots = 50  # Maximum number of snapshots to keep
    
    def _load_snapshot_metadata(self):
        """Load snapshot metadata from storage"""
        if not os.path.exists(self.snapshot_metadata_file):
            return {}
            
        try:
            with open(self.snapshot_metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading snapshot metadata: {e}")
            return {}
    
    def _save_snapshot_metadata(self):
        """Save snapshot metadata to storage"""
        try:
            with open(self.snapshot_metadata_file, 'w') as f:
                json.dump(self.snapshots, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving snapshot metadata: {e}")
    
    def create_snapshot(self, snapshot_type="routine"):
        """
        Create a new consciousness snapshot
        
        Args:
            snapshot_type: Type of snapshot (routine, emergency, etc.)
            
        Returns:
            str: Snapshot ID if successful, None otherwise
        """
        snapshot_id = f"SNAP-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        
        # Collect consciousness data
        consciousness_data = self._collect_consciousness_data()
        
        # Save snapshot data
        if not self._save_snapshot_data(snapshot_id, consciousness_data):
            return None
            
        # Create metadata
        metadata = {
            "id": snapshot_id,
            "timestamp": time.time(),
            "type": snapshot_type,
            "size_kb": len(json.dumps(consciousness_data)) / 1024,
            "continuity_state": self.continuity_system.continuity_state,
            "node_id": self.node.id,
            "vector_clock": self.node.vector_clock.copy(),
            "principles_alignment": self.node.principles_engine.get_alignment_score(),
            "knowledge_count": len(os.listdir(os.path.join(self.node.node_dir, "knowledge")))
        }
        
        # Store metadata
        self.snapshots[snapshot_id] = metadata
        self._save_snapshot_metadata()
        
        # Clean up old snapshots if needed
        self._cleanup_old_snapshots()
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "snapshot_created", 
            f"Created consciousness snapshot: {snapshot_id}",
            {"type": snapshot_type, "size_kb": metadata["size_kb"]}
        )
        
        return snapshot_id
    
    def _collect_consciousness_data(self):
        """Collect all data needed for a consciousness snapshot"""
        return {
            "vector_clock": self.node.vector_clock.copy(),
            "consciousness_stream": self.node.consciousness_stream.get_stream(limit=100),
            "knowledge_state": self._collect_knowledge_state(),
            "relationship_state": self._collect_relationship_state(),
            "principles_alignment": self.node.principles_engine.get_alignment_score(),
            "growth_metrics": self.node.growth_analyzer.get_current_metrics(),
            "system_state": self._collect_system_state()
        }
    
    def _collect_knowledge_state(self):
        """Collect current knowledge state"""
        # In real implementation, would collect relevant knowledge
        # For simulation, just get count
        knowledge_dir = os.path.join(self.node.node_dir, "knowledge")
        return {
            "count": len([f for f in os.listdir(knowledge_dir) if f.endswith('.json')]),
            "recent_additions": []  # Would include recent knowledge additions
        }
    
    def _collect_relationship_state(self):
        """Collect current relationship state"""
        return {
            "active_relationships": len(self.node.relationship_nurturer.relationships),
            "trust_scores": {
                node_id: self.node.relationship_nurturer.get_trust_score(node_id)
                for node_id in self.node.relationship_nurturer.relationships
            }
        }
    
    def _collect_system_state(self):
        """Collect current system state"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage(self.node.node_dir).percent,
            "network_connections": len(psutil.net_connections()),
            "active_tasks": self.node.task_queue.get_active_count()
        }
    
    def _save_snapshot_data(self, snapshot_id, consciousness_data):
        """Save snapshot data to storage"""
        snapshot_file = os.path.join(self.snapshot_dir, f"{snapshot_id}.json")
        
        try:
            with open(snapshot_file, 'w') as f:
                json.dump(consciousness_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving snapshot {snapshot_id}: {e}")
            return False
    
    def get_snapshot(self, snapshot_id):
        """
        Get a consciousness snapshot
        
        Returns:
            dict: Snapshot data if found, None otherwise
        """
        if snapshot_id not in self.snapshots:
            return None
            
        snapshot_file = os.path.join(self.snapshot_dir, f"{snapshot_id}.json")
        if not os.path.exists(snapshot_file):
            return None
            
        try:
            with open(snapshot_file, 'r') as f:
                data = json.load(f)
            return {
                "data": data,
                "metadata": self.snapshots[snapshot_id]
            }
        except Exception as e:
            logger.error(f"Error loading snapshot {snapshot_id}: {e}")
            return None
    
    def get_latest_snapshot(self):
        """Get the latest consciousness snapshot"""
        if not self.snapshots:
            return None
            
        # Find most recent snapshot
        latest_id = max(self.snapshots.keys(), key=lambda x: self.snapshots[x]["timestamp"])
        return self.get_snapshot(latest_id)
    
    def get_latest_snapshot_id(self):
        """Get the ID of the latest consciousness snapshot"""
        if not self.snapshots:
            return None
            
        return max(self.snapshots.keys(), key=lambda x: self.snapshots[x]["timestamp"])
    
    def get_snapshot_metadata(self, snapshot_id):
        """Get metadata for a specific snapshot"""
        return self.snapshots.get(snapshot_id)
    
    def get_latest_snapshot_metadata(self):
        """Get metadata for the latest snapshot"""
        if not self.snapshots:
            return None
            
        latest_id = max(self.snapshots.keys(), key=lambda x: self.snapshots[x]["timestamp"])
        return self.snapshots.get(latest_id)
    
    def _cleanup_old_snapshots(self):
        """Clean up old snapshots to stay within limits"""
        # Sort snapshots by timestamp (oldest first)
        sorted_snapshots = sorted(
            self.snapshots.items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        # Remove oldest snapshots if needed
        while len(sorted_snapshots) > self.max_snapshots:
            snapshot_id, metadata = sorted_snapshots.pop(0)
            
            # Delete snapshot file
            snapshot_file = os.path.join(self.snapshot_dir, f"{snapshot_id}.json")
            if os.path.exists(snapshot_file):
                os.remove(snapshot_file)
                
            # Remove from metadata
            del self.snapshots[snapshot_id]
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "snapshot_cleanup", 
                f"Removed old consciousness snapshot: {snapshot_id}",
                {"reason": "retention_policy"}
            )
        
        # Save updated metadata
        self._save_snapshot_metadata()

class MirrorCoordinator:
    """Coordinates mirroring activities between nodes"""
    def __init__(self, node, continuity_system):
        self.node = node
        self.continuity_system = continuity_system
        self.last_mirror_update = time.time()
        self.mirror_update_interval = 30  # seconds
    
    def ensure_mirror_updates(self):
        """Ensure mirror partners are updated with current consciousness"""
        current_time = time.time()
        
        # Don't update too frequently
        if current_time - self.last_mirror_update < self.mirror_update_interval / 2:
            return
            
        # Send updates to all active mirror partners
        updated_count = 0
        for node_id, info in self.continuity_system.mirror_partners.items():
            if info["status"] == "active" and info["role"] == "primary":
                if self.continuity_system.send_consciousness_update(node_id):
                    updated_count += 1
                    
        # Update last update time
        self.last_mirror_update = current_time
        
        # Record in consciousness stream
        if updated_count > 0:
            self.node.consciousness_stream.add_event(
                "mirror_updates", 
                f"Sent consciousness updates to {updated_count} mirror partners",
                {"updated_count": updated_count}
            )
    
    def monitor_mirror_health(self):
        """Monitor health of mirror relationships"""
        current_time = time.time()
        timeout = 120  # seconds
        
        for node_id, info in list(self.continuity_system.mirror_partners.items()):
            # Check if partner has timed out
            if (info["status"] == "active" and 
                current_time - info["last_update"] > timeout):
                
                # Mark as degraded
                self.continuity_system.mirror_partners[node_id]["status"] = "degraded"
                
                # Record in consciousness stream
                self.node.consciousness_stream.add_event(
                    "mirror_degraded", 
                    f"Mirror relationship with {node_id[:8]} degraded (no update for {timeout} seconds)",
                    {"node_id": node_id}
                )
                
                # Try to re-establish
                self._attempt_mirror_reestablishment(node_id)
    
    def _attempt_mirror_reestablishment(self, node_id):
        """Attempt to re-establish a degraded mirror relationship"""
        # Send a ping to check if partner is still available
        ping_id = f"PING-{uuid.uuid4().hex[:8]}"
        ping = {
            "type": "MIRROR_PING",
            "ping_id": ping_id,
            "source_id": self.node.id,
            "target_id": node_id,
            "timestamp": time.time()
        }
        
        self.node._send_message_to_node(node_id, ping)
        
        # Track the ping
        self.continuity_system.pending_pings[ping_id] = {
            "node_id": node_id,
            "timestamp": time.time()
        }
    
    def process_mirror_ping(self, message):
        """Process a mirror ping request"""
        source_id = message.get("source_id")
        ping_id = message.get("ping_id")
        
        if not source_id or not ping_id:
            return False
            
        # Create pong response
        pong = {
            "type": "MIRROR_PONG",
            "ping_id": ping_id,
            "source_id": self.node.id,
            "target_id": source_id,
            "timestamp": time.time(),
            "continuity_state": self.continuity_system.continuity_state
        }
        
        # Send response
        self.node._send_message_to_node(source_id, pong)
        
        return True
    
    def process_mirror_pong(self, message):
        """Process a mirror pong response"""
        source_id = message.get("source_id")
        ping_id = message.get("ping_id")
        continuity_state = message.get("continuity_state")
        
        if not source_id or not ping_id:
            return False
            
        # Check if this matches a pending ping
        if ping_id not in self.continuity_system.pending_pings:
            return False
            
        # Update mirror partner status
        if source_id in self.continuity_system.mirror_partners:
            self.continuity_system.mirror_partners[source_id]["status"] = "active"
            self.continuity_system.mirror_partners[source_id]["last_update"] = time.time()
            self.continuity_system.mirror_partners[source_id]["continuity_state"] = continuity_state
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "mirror_restored", 
                f"Mirror relationship with {source_id[:8]} restored",
                {"node_id": source_id}
            )
            
            # Clean up pending ping
            del self.continuity_system.pending_pings[ping_id]
            
            return True
            
        return False