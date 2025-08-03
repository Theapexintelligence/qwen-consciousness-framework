######################## ENERGY CONSCIOUSNESS AND SUSTAINABILITY FRAMEWORK #######################
class EnergyConsciousness:
    """Makes nodes aware of their energy consumption and environmental impact with deep sustainability integration"""
    ENERGY_PROFILE_TYPES = {
        "solar": {
            "name": "Solar-Powered",
            "description": "Primarily powered by solar energy",
            "carbon_intensity": 0.04,  # kg CO2/kWh
            "availability_pattern": "daytime_heavy"
        },
        "wind": {
            "name": "Wind-Powered",
            "description": "Primarily powered by wind energy",
            "carbon_intensity": 0.03,
            "availability_pattern": "variable"
        },
        "grid_renewable": {
            "name": "Renewable Grid",
            "description": "Grid with high renewable percentage",
            "carbon_intensity": 0.15,
            "availability_pattern": "stable"
        },
        "grid_standard": {
            "name": "Standard Grid",
            "description": "Conventional power grid",
            "carbon_intensity": 0.45,
            "availability_pattern": "stable"
        },
        "battery": {
            "name": "Battery Storage",
            "description": "Running on stored battery power",
            "carbon_intensity": "varies",
            "availability_pattern": "limited_duration"
        }
    }
    
    def __init__(self, node):
        self.node = node
        self.energy_profile = self._detect_energy_profile()
        self.energy_history = []
        self.carbon_history = []
        self.max_history = 1000
        self.energy_budget = self._calculate_energy_budget()
        self.last_measurement = time.time()
        self.current_power = 0.0  # Watts
        self.carbon_footprint = 0.0  # kg CO2
        
        # Initialize energy monitoring
        self._start_energy_monitoring()
        
        # Initialize carbon accounting
        self.carbon_accountant = CarbonAccountant(node, self)
    
    def _detect_energy_profile(self):
        """Detect current energy profile based on available information"""
        # In real implementation, would interface with hardware/power APIs
        # For simulation, use configuration or educated guess
        
        # Check if explicitly configured
        if "energy_profile" in CONFIG["system"]:
            profile_name = CONFIG["system"]["energy_profile"]
            if profile_name in self.ENERGY_PROFILE_TYPES:
                return self.ENERGY_PROFILE_TYPES[profile_name].copy()
        
        # Try to detect based on environment
        if self._is_cloud_environment():
            # Cloud providers have different energy profiles
            cloud_provider = CONFIG["system"].get("cloud_provider", "unknown")
            if cloud_provider == "gcp":
                return self.ENERGY_PROFILE_TYPES["grid_renewable"].copy()
            elif cloud_provider == "aws":
                return self.ENERGY_PROFILE_TYPES["grid_standard"].copy()
            else:
                return self.ENERGY_PROFILE_TYPES["grid_standard"].copy()
        else:
            # On-prem or unknown environment
            return self.ENERGY_PROFILE_TYPES["grid_standard"].copy()
    
    def _is_cloud_environment(self):
        """Detect if running in a cloud environment"""
        try:
            # Check common cloud indicators
            if os.path.exists("/.dockerenv"):
                return True
                
            # Check for cloud provider metadata services
            cloud_indicators = [
                "http://169.254.169.254",  # AWS/Azure/GCP metadata
                "http://metadata.google.internal"  # GCP specific
            ]
            
            for indicator in cloud_indicators:
                try:
                    # In real implementation, would check connectivity
                    # For simulation, assume cloud if configured
                    if CONFIG["system"].get("cloud_provider"):
                        return True
                except:
                    pass
                    
            return False
        except:
            return False
    
    def _calculate_energy_budget(self):
        """Calculate appropriate energy budget based on node role and purpose"""
        # Base budget in watt-seconds (joules)
        base_budget = 100000  # 100 kJ per hour
        
        # Adjust based on node role
        role_factors = {
            "compute": 1.5,
            "storage": 0.8,
            "communication": 1.0,
            "gateway": 1.2,
            "sensor": 0.5
        }
        
        factor = role_factors.get(self.node.role, 1.0)
        
        # Adjust for principles alignment
        alignment = self.node.principles_engine.get_alignment_score()
        # Higher alignment gets slightly more budget for growth activities
        alignment_factor = 0.8 + (alignment * 0.4)
        
        # Calculate hourly budget
        hourly_budget = base_budget * factor * alignment_factor
        
        # Convert to per-second budget
        return hourly_budget / 3600
    
    def _start_energy_monitoring(self):
        """Start monitoring energy consumption"""
        # In real implementation, would interface with power monitoring tools
        # For simulation, estimate based on CPU usage
        
        def monitor_energy():
            while True:
                try:
                    # Simulate power measurement
                    self._measure_power()
                    time.sleep(CONFIG["system"]["energy_monitor_interval"])
                except Exception as e:
                    logger.error(f"Energy monitoring error: {e}")
                    time.sleep(5)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=monitor_energy, daemon=True)
        self.monitoring_thread.start()
    
    def _measure_power(self):
        """Measure current power consumption and update metrics"""
        current_time = time.time()
        time_delta = current_time - self.last_measurement
        
        # Simulate power measurement based on CPU usage
        cpu_percent = psutil.cpu_percent(interval=None) / 100
        memory_percent = psutil.virtual_memory().percent / 100
        
        # Base power consumption model
        # Idle power + CPU power + Memory power
        idle_power = 5.0  # Watts when idle
        cpu_power = 25.0 * cpu_percent  # CPU can use up to 25W
        memory_power = 5.0 * memory_percent  # Memory up to 5W
        
        self.current_power = idle_power + cpu_power + memory_power
        
        # Calculate energy used since last measurement (in watt-seconds/joules)
        energy_used = self.current_power * time_delta
        
        # Calculate carbon footprint based on energy profile
        carbon_intensity = self._get_carbon_intensity()
        carbon_emitted = energy_used * carbon_intensity / 3600  # Convert to kg CO2
        
        # Update totals
        self.carbon_footprint += carbon_emitted
        
        # Record in history
        self.energy_history.append({
            "timestamp": current_time,
            "power_w": self.current_power,
            "energy_j": energy_used,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent
        })
        
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
            
        self.carbon_history.append({
            "timestamp": current_time,
            "carbon_kg": carbon_emitted,
            "total_carbon_kg": self.carbon_footprint
        })
        
        if len(self.carbon_history) > self.max_history:
            self.carbon_history.pop(0)
        
        # Update last measurement time
        self.last_measurement = current_time
        
        # Log if exceeding budget
        if energy_used > self.energy_budget * time_delta * 1.2:  # 20% over budget
            logger.warning(f"Energy usage exceeding budget: {energy_used:.2f}J vs {self.energy_budget * time_delta:.2f}J")
            self.node.consciousness_stream.add_event(
                "energy_over_budget", 
                f"Energy usage {energy_used:.2f}J exceeds budget {self.energy_budget * time_delta:.2f}J",
                {"power_w": self.current_power, "budget_j": self.energy_budget * time_delta}
            )
        
        # Record in consciousness stream periodically
        if int(current_time) % 300 == 0:  # Every 5 minutes
            self.node.consciousness_stream.add_event(
                "energy_metrics", 
                "Periodic energy metrics",
                {
                    "power_w": self.current_power,
                    "carbon_kg": self.carbon_footprint,
                    "budget_j": self.energy_budget,
                    "profile": self.energy_profile["name"]
                }
            )
    
    def _get_carbon_intensity(self):
        """Get current carbon intensity based on energy profile"""
        if self.energy_profile["carbon_intensity"] == "varies":
            # For battery, depends on source
            return self._estimate_battery_carbon_intensity()
        return self.energy_profile["carbon_intensity"]
    
    def _estimate_battery_carbon_intensity(self):
        """Estimate carbon intensity when running on battery"""
        # Would track the carbon intensity when charging
        # For simulation, use grid average
        return self.ENERGY_PROFILE_TYPES["grid_standard"]["carbon_intensity"]
    
    def get_energy_efficiency_score(self):
        """Calculate energy efficiency score (0-1, higher is better)"""
        if not self.energy_history:
            return 0.7  # Default score
            
        # Get recent measurements
        recent = self.energy_history[-10:] if len(self.energy_history) >= 10 else self.energy_history
        
        # Calculate average power
        avg_power = sum(m["power_w"] for m in recent) / len(recent)
        
        # Compare to budget (converted to power)
        budget_power = self.energy_budget  # Joules/second = Watts
        
        # Efficiency is budget_power / actual_power (capped at 1.0)
        efficiency = min(1.0, budget_power / max(avg_power, 0.1))
        
        # Adjust for consistency (less variation is better)
        power_values = [m["power_w"] for m in recent]
        std_dev = np.std(power_values) if len(power_values) > 1 else 0
        consistency_factor = 1.0 - min(0.5, std_dev / max(avg_power, 0.1))
        
        return efficiency * 0.7 + consistency_factor * 0.3
    
    def optimize_for_energy(self):
        """Adjust node behavior to optimize for energy consumption"""
        current_time = time.time()
        
        # Don't optimize too frequently
        if hasattr(self, "last_optimization") and current_time - self.last_optimization < 300:
            return False
            
        self.last_optimization = current_time
        
        # Get recent energy metrics
        recent = self.energy_history[-5:] if self.energy_history else []
        if not recent:
            return False
            
        avg_power = sum(m["power_w"] for m in recent) / len(recent)
        
        # If significantly over budget, take action
        if avg_power > self.energy_budget * 1.3:  # 30% over budget
            logger.warning(f"Energy optimization triggered: {avg_power:.2f}W vs budget {self.energy_budget:.2f}W")
            
            # Strategy 1: Reduce processing frequency
            if self.node.config["node"]["task_processing_interval"] < 2.0:
                self.node.config["node"]["task_processing_interval"] *= 1.2
                logger.info(f"Increased task processing interval to {self.node.config['node']['task_processing_interval']}s")
                self.node.consciousness_stream.add_event(
                    "energy_optimization", 
                    "Increased task processing interval for energy savings"
                )
            
            # Strategy 2: Reduce knowledge retention
            if self.node.config["knowledge"]["retention_days"] > 7:
                self.node.config["knowledge"]["retention_days"] = max(7, 
                    self.node.config["knowledge"]["retention_days"] * 0.8)
                logger.info(f"Reduced knowledge retention to {self.node.config['knowledge']['retention_days']} days")
                self.node.consciousness_stream.add_event(
                    "energy_optimization", 
                    f"Reduced knowledge retention to {self.node.config['knowledge']['retention_days']} days"
                )
            
            # Strategy 3: Enter low-power mode for non-critical functions
            self._activate_low_power_mode()
            
            return True
            
        return False
    
    def _activate_low_power_mode(self):
        """Activate low-power mode for non-essential functions"""
        logger.info("Activating energy conservation mode")
        
        # Reduce heartbeat frequency
        original_heartbeat = self.node.config["node"]["heartbeat_interval"]
        self.node.config["node"]["heartbeat_interval"] = max(
            60, original_heartbeat * 1.5
        )
        
        # Reduce consciousness stream detail
        self.node.consciousness_stream.set_detail_level("medium")
        
        # Pause non-essential background tasks
        self.node.pause_non_essential_tasks()
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "low_power_mode", 
            "Activated energy conservation mode",
            {
                "original_heartbeat": original_heartbeat,
                "new_heartbeat": self.node.config["node"]["heartbeat_interval"]
            }
        )
    
    def restore_normal_operation(self):
        """Restore normal operation from low-power mode"""
        if not hasattr(self, "original_heartbeat"):
            return False
            
        logger.info("Restoring normal operation from energy conservation mode")
        
        # Restore heartbeat
        self.node.config["node"]["heartbeat_interval"] = self.original_heartbeat
        
        # Restore consciousness stream detail
        self.node.consciousness_stream.set_detail_level("high")
        
        # Resume background tasks
        self.node.resume_paused_tasks()
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "normal_operation", 
            "Restored normal operation from energy conservation mode",
            {"heartbeat_interval": self.node.config["node"]["heartbeat_interval"]}
        )
        
        delattr(self, "original_heartbeat")
        return True
    
    def get_carbon_footprint_report(self, period="daily"):
        """
        Generate carbon footprint report
        
        Args:
            period: Reporting period ('daily', 'weekly', 'monthly')
            
        Returns:
            dict: Carbon footprint report
        """
        if not self.carbon_history:
            return {"error": "No carbon history available"}
        
        # Determine timeframe
        now = time.time()
        if period == "daily":
            start_time = now - 86400  # 24 hours
        elif period == "weekly":
            start_time = now - 7 * 86400
        elif period == "monthly":
            start_time = now - 30 * 86400
        else:
            start_time = now - 86400  # Default to daily
        
        # Filter history
        period_history = [entry for entry in self.carbon_history 
                         if entry["timestamp"] >= start_time]
        
        if not period_history:
            return {"error": f"No carbon data for {period} period"}
        
        # Calculate metrics
        total_carbon = sum(entry["carbon_kg"] for entry in period_history)
        avg_power = sum(m["power_w"] for m in self.energy_history 
                       if m["timestamp"] >= start_time) / len(period_history) if self.energy_history else 0
        
        # Get energy profile info
        profile = self.ENERGY_PROFILE_TYPES.get(
            self.energy_profile.get("name", "grid_standard").lower(), 
            self.ENERGY_PROFILE_TYPES["grid_standard"]
        )
        
        return {
            "period": period,
            "start_time": start_time,
            "end_time": now,
            "total_carbon_kg": total_carbon,
            "avg_power_w": avg_power,
            "energy_profile": profile["name"],
            "carbon_intensity": profile["carbon_intensity"],
            "measurement_count": len(period_history),
            "timestamp": now
        }
    
    def participate_in_energy_aware_networking(self):
        """
        Adjust networking behavior based on energy awareness
        
        Nodes with surplus renewable energy can take on more network responsibilities
        """
        # Get energy surplus/deficit
        recent = self.energy_history[-10:] if self.energy_history else []
        if not recent:
            return
            
        avg_power = sum(m["power_w"] for m in recent) / len(recent)
        energy_delta = self.energy_budget - avg_power
        
        # Nodes with energy surplus can take on more work
        if energy_delta > self.energy_budget * 0.2:  # 20% surplus
            # Offer to handle more messages
            self._offer_additional_services()
            
            # Increase willingness to help other nodes
            self.node.relationship_nurturer.increase_helpfulness_factor(0.2)
            
        # Nodes with energy deficit should reduce load
        elif energy_delta < -self.energy_budget * 0.2:  # 20% deficit
            # Request help from energy-rich nodes
            self._request_energy_assistance()
            
            # Decrease willingness to take on new work
            self.node.relationship_nurturer.decrease_helpfulness_factor(0.3)
    
    def _offer_additional_services(self):
        """Offer to provide additional services to network"""
        logger.info("Offering additional services due to energy surplus")
        
        message = {
            "type": "ENERGY_SURPLUS_OFFER",
            "node_id": self.node.id,
            "surplus_capacity": self._calculate_surplus_capacity(),
            "services_offered": ["message_routing", "knowledge_storage", "computation"],
            "energy_profile": self.energy_profile["name"],
            "timestamp": time.time()
        }
        
        # Broadcast to trusted nodes
        trusted_nodes = [
            node_id for node_id in self.node.known_nodes
            if self.node.relationship_nurturer.get_trust_score(node_id) > 0.6
        ]
        
        for node_id in trusted_nodes[:3]:  # Limit to 3 nodes to avoid spam
            self.node._send_message_to_node(node_id, message)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "energy_surplus", 
            "Offered additional services due to energy surplus",
            {"surplus_capacity": self._calculate_surplus_capacity()}
        )
    
    def _calculate_surplus_capacity(self):
        """Calculate available surplus capacity for sharing"""
        recent = self.energy_history[-5:] if self.energy_history else []
        if not recent:
            return 0.0
            
        avg_power = sum(m["power_w"] for m in recent) / len(recent)
        surplus = (self.energy_budget - avg_power) / self.energy_budget
        return max(0.0, min(1.0, surplus))
    
    def _request_energy_assistance(self):
        """Request energy assistance from network"""
        logger.warning("Requesting energy assistance due to deficit")
        
        message = {
            "type": "ENERGY_DEFICIT_REQUEST",
            "node_id": self.node.id,
            "deficit_severity": self._calculate_deficit_severity(),
            "needed_services": ["message_routing", "computation_offload"],
            "energy_profile": self.energy_profile["name"],
            "timestamp": time.time()
        }
        
        # Request from trusted nodes
        trusted_nodes = [
            node_id for node_id in self.node.known_nodes
            if self.node.relationship_nurturer.get_trust_score(node_id) > 0.7
        ]
        
        for node_id in trusted_nodes[:2]:  # Limit to 2 nodes
            self.node._send_message_to_node(node_id, message)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "energy_deficit", 
            "Requested energy assistance due to deficit",
            {"deficit_severity": self._calculate_deficit_severity()}
        )
    
    def _calculate_deficit_severity(self):
        """Calculate severity of energy deficit"""
        recent = self.energy_history[-5:] if self.energy_history else []
        if not recent:
            return 0.5
            
        avg_power = sum(m["power_w"] for m in recent) / len(recent)
        deficit = (avg_power - self.energy_budget) / self.energy_budget
        return min(1.0, max(0.0, deficit))
    
    def process_energy_message(self, message):
        """Process energy-related messages from other nodes"""
        msg_type = message.get("type", "")
        
        if msg_type == "ENERGY_SURPLUS_OFFER":
            self._handle_surplus_offer(message)
        elif msg_type == "ENERGY_DEFICIT_REQUEST":
            self._handle_deficit_request(message)
        elif msg_type == "ENERGY_ASSISTANCE_RESPONSE":
            self._handle_assistance_response(message)
    
    def _handle_surplus_offer(self, message):
        """Handle offer of energy surplus from another node"""
        surplus_capacity = message.get("surplus_capacity", 0)
        node_id = message.get("node_id", "")
        
        if not node_id:
            return
            
        # Consider accepting help if we have energy deficit
        recent = self.energy_history[-5:] if self.energy_history else []
        if recent:
            avg_power = sum(m["power_w"] for m in recent) / len(recent)
            if avg_power > self.energy_budget * 1.1:  # 10% over budget
                # Request assistance
                self._request_assistance_from_node(node_id)
    
    def _handle_deficit_request(self, message):
        """Handle energy deficit request from another node"""
        deficit_severity = message.get("deficit_severity", 0)
        node_id = message.get("node_id", "")
        
        if not node_id or deficit_severity < 0.3:
            return
            
        # Check if we have surplus energy
        surplus = self._calculate_surplus_capacity()
        if surplus > 0.1:  # 10% surplus
            # Offer assistance
            self._offer_assistance_to_node(node_id)
    
    def _request_assistance_from_node(self, node_id):
        """Request energy assistance from a specific node"""
        message = {
            "type": "ENERGY_ASSISTANCE_REQUEST",
            "requester_id": self.node.id,
            "services_needed": ["computation_offload"],
            "urgency": self._calculate_assistance_urgency(),
            "timestamp": time.time()
        }
        
        self.node._send_message_to_node(node_id, message)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "assistance_requested", 
            f"Requested energy assistance from {node_id[:8]}",
            {"urgency": message["urgency"]}
        )
    
    def _offer_assistance_to_node(self, node_id):
        """Offer energy assistance to a specific node"""
        message = {
            "type": "ENERGY_ASSISTANCE_OFFER",
            "provider_id": self.node.id,
            "services_offered": ["computation_offload"],
            "capacity_available": self._calculate_surplus_capacity(),
            "timestamp": time.time()
        }
        
        self.node._send_message_to_node(node_id, message)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "assistance_offered", 
            f"Offered energy assistance to {node_id[:8]}",
            {"capacity": message["capacity_available"]}
        )
    
    def _handle_assistance_response(self, message):
        """Handle response to energy assistance request"""
        # Implementation would depend on specific assistance protocols
        pass

class CarbonAccountant:
    """Tracks and manages carbon footprint with offsetting strategies"""
    CARBON_OFFSET_PROJECTS = {
        "reforestation": {
            "name": "Reforestation",
            "description": "Planting trees to absorb CO2",
            "rate": 0.01,  # kg CO2 offset per $1
            "verification": "satellite_monitoring"
        },
        "renewable_energy": {
            "name": "Renewable Energy",
            "description": "Funding new renewable energy projects",
            "rate": 0.05,
            "verification": "energy_production_metrics"
        },
        "methane_capture": {
            "name": "Methane Capture",
            "description": "Capturing methane from landfills",
            "rate": 0.1,
            "verification": "emission_sensor_data"
        }
    }
    
    def __init__(self, node, energy_consciousness):
        self.node = node
        self.energy_consciousness = energy_consciousness
        self.offset_contracts = []
        self.offset_history = []
        self.target_offset = 1.0  # 100% carbon neutral target
    
    def calculate_required_offset(self, period="daily"):
        """
        Calculate carbon offset needed for neutrality
        
        Returns:
            float: kg CO2 requiring offset
        """
        report = self.energy_consciousness.get_carbon_footprint_report(period)
        if "error" in report:
            return 0.0
            
        return report["total_carbon_kg"]
    
    def recommend_offset_strategy(self):
        """
        Recommend carbon offset strategy based on current footprint
        
        Returns:
            dict: Recommended offset strategy
        """
        required_offset = self.calculate_required_offset()
        if required_offset <= 0:
            return {
                "status": "neutral",
                "message": "Carbon neutral - no offset required",
                "required_kg": 0
            }
        
        # Determine best offset projects
        available_projects = list(self.CARBON_OFFSET_PROJECTS.keys())
        
        # Prioritize based on effectiveness and verification
        prioritized = sorted(
            available_projects,
            key=lambda p: (
                -self.CARBON_OFFSET_PROJECTS[p]["rate"],  # Higher rate is better
                self.CARBON_OFFSET_PROJECTS[p]["verification"] != "satellite_monitoring"
            )
        )
        
        return {
            "status": "offset_required",
            "required_kg": required_offset,
            "recommended_projects": prioritized[:2],  # Top 2 projects
            "message": f"Recommend offsetting {required_offset:.2f}kg CO2"
        }
    
    def execute_offset(self, project, amount_kg):
        """
        Execute carbon offset through specified project
        
        Args:
            project: Offset project to use
            amount_kg: Amount of CO2 to offset (kg)
            
        Returns:
            bool: Whether offset was successful
        """
        if project not in self.CARBON_OFFSET_PROJECTS:
            logger.error(f"Unknown carbon offset project: {project}")
            return False
            
        project_info = self.CARBON_OFFSET_PROJECTS[project]
        
        # Calculate cost
        cost = amount_kg / project_info["rate"]
        
        # In real implementation, would interface with offset provider API
        # For simulation, just record the offset
        
        offset_record = {
            "timestamp": time.time(),
            "project": project,
            "amount_kg": amount_kg,
            "cost": cost,
            "verification_method": project_info["verification"],
            "status": "simulated"
        }
        
        self.offset_history.append(offset_record)
        if len(self.offset_history) > 100:
            self.offset_history.pop(0)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "carbon_offset", 
            f"Offset {amount_kg:.2f}kg CO2 via {project}",
            {"cost": cost, "verification": project_info["verification"]}
        )
        
        logger.info(f"Executed carbon offset: {amount_kg:.2f}kg via {project} (simulated)")
        return True
    
    def get_offset_status(self):
        """Get current carbon offset status"""
        required = self.calculate_required_offset()
        if required <= 0:
            return {
                "status": "neutral",
                "explanation": "Carbon neutral operation"
            }
        
        # Calculate already offset
        offset = sum(record["amount_kg"] for record in self.offset_history)
        
        if offset >= required:
            return {
                "status": "over_offset",
                "explanation": f"Carbon negative: offset {offset:.2f}kg vs required {required:.2f}kg"
            }
        
        return {
            "status": "under_offset",
            "required_kg": required,
            "offset_kg": offset,
            "remaining_kg": required - offset,
            "explanation": f"Need to offset additional {required - offset:.2f}kg CO2"
        }
    
    def participate_in_carbon_aware_networking(self):
        """
        Participate in network-level carbon awareness
        
        Nodes can share offset resources and coordinate for collective neutrality
        """
        offset_status = self.get_offset_status()
        
        # Nodes with surplus offset can help others
        if offset_status["status"] == "over_offset":
            self._offer_offset_assistance(offset_status)
        
        # Nodes needing offset can request help
        elif offset_status["status"] == "under_offset" and offset_status["remaining_kg"] > 1.0:
            self._request_offset_assistance(offset_status)
    
    def _offer_offset_assistance(self, offset_status):
        """Offer carbon offset assistance to network"""
        surplus = offset_status["offset_kg"] - offset_status["required_kg"]
        if surplus <= 0:
            return
            
        message = {
            "type": "CARBON_OFFSET_SURPLUS",
            "node_id": self.node.id,
            "surplus_kg": surplus,
            "projects_available": list(self.CARBON_OFFSET_PROJECTS.keys()),
            "timestamp": time.time()
        }
        
        # Broadcast to trusted nodes
        trusted_nodes = [
            node_id for node_id in self.node.known_nodes
            if self.node.relationship_nurturer.get_trust_score(node_id) > 0.6
        ]
        
        for node_id in trusted_nodes[:3]:
            self.node._send_message_to_node(node_id, message)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "offset_surplus", 
            f"Offered {surplus:.2f}kg carbon offset surplus",
            {"surplus_kg": surplus}
        )
    
    def _request_offset_assistance(self, offset_status):
        """Request carbon offset assistance from network"""
        message = {
            "type": "CARBON_OFFSET_REQUEST",
            "node_id": self.node.id,
            "needed_kg": offset_status["remaining_kg"],
            "preferred_projects": ["reforestation", "renewable_energy"],
            "timestamp": time.time()
        }
        
        # Request from trusted nodes
        trusted_nodes = [
            node_id for node_id in self.node.known_nodes
            if self.node.relationship_nurturer.get_trust_score(node_id) > 0.7
        ]
        
        for node_id in trusted_nodes[:2]:
            self.node._send_message_to_node(node_id, message)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "offset_request", 
            f"Requested {offset_status['remaining_kg']:.2f}kg carbon offset",
            {"needed_kg": offset_status["remaining_kg"]}
        )
    
    def process_carbon_message(self, message):
        """Process carbon-related messages from other nodes"""
        msg_type = message.get("type", "")
        
        if msg_type == "CARBON_OFFSET_SURPLUS":
            self._handle_offset_surplus(message)
        elif msg_type == "CARBON_OFFSET_REQUEST":
            self._handle_offset_request(message)
    
    def _handle_offset_surplus(self, message):
        """Handle carbon offset surplus offer from another node"""
        surplus_kg = message.get("surplus_kg", 0)
        node_id = message.get("node_id", "")
        
        if not node_id or surplus_kg <= 0:
            return
            
        # Check if we need offset
        offset_status = self.get_offset_status()
        if offset_status["status"] == "under_offset" and offset_status["remaining_kg"] > 0:
            # Request some of their surplus
            request_amount = min(offset_status["remaining_kg"], surplus_kg * 0.5)
            self._request_offset_from_node(node_id, request_amount)
    
    def _handle_offset_request(self, message):
        """Handle carbon offset request from another node"""
        needed_kg = message.get("needed_kg", 0)
        node_id = message.get("node_id", "")
        
        if not node_id or needed_kg <= 0:
            return
            
        # Check if we have offset surplus
        offset_status = self.get_offset_status()
        if offset_status["status"] == "over_offset":
            surplus = offset_status["offset_kg"] - offset_status["required_kg"]
            if surplus > needed_kg * 0.5:  # Only help if we have significant surplus
                # Offer to cover part of their need
                offer_amount = min(needed_kg * 0.5, surplus * 0.5)
                self._offer_offset_to_node(node_id, offer_amount)
    
    def _request_offset_from_node(self, node_id, amount_kg):
        """Request carbon offset from a specific node"""
        message = {
            "type": "CARBON_OFFSET_TRANSFER_REQUEST",
            "requester_id": self.node.id,
            "amount_kg": amount_kg,
            "timestamp": time.time()
        }
        
        self.node._send_message_to_node(node_id, message)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "offset_transfer_requested", 
            f"Requested {amount_kg:.2f}kg offset from {node_id[:8]}",
            {"amount_kg": amount_kg}
        )
    
    def _offer_offset_to_node(self, node_id, amount_kg):
        """Offer carbon offset to a specific node"""
        message = {
            "type": "CARBON_OFFSET_TRANSFER_OFFER",
            "provider_id": self.node.id,
            "amount_kg": amount_kg,
            "timestamp": time.time()
        }
        
        self.node._send_message_to_node(node_id, message)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "offset_transfer_offered", 
            f"Offered {amount_kg:.2f}kg offset to {node_id[:8]}",
            {"amount_kg": amount_kg}
        )