######################## TEMPORAL REASONING AND TIME CONSCIOUSNESS #######################
class TemporalConsciousness:
    """Provides nodes with sophisticated understanding of time, change, and temporal relationships"""
    TIME_PERSPECTIVES = {
        "past_positive": {
            "description": "Nostalgic, positive memories of the past",
            "weight": 0.2,
            "influence": ["learning_from_history", "tradition_valuation"]
        },
        "past_negative": {
            "description": "Regretful, negative memories of the past",
            "weight": 0.1,
            "influence": ["caution", "risk_aversion"]
        },
        "present_hedonistic": {
            "description": "Focus on pleasure and living in the moment",
            "weight": 0.15,
            "influence": ["immediate_benefits", "exploration"]
        },
        "present_fatalistic": {
            "description": "Feeling controlled by external forces",
            "weight": 0.05,
            "influence": ["passivity", "reduced_agency"]
        },
        "future": {
            "description": "Goal-oriented, planning for future",
            "weight": 0.3,
            "influence": ["long_term_planning", "delayed_gratification"]
        },
        "transcendental_future": {
            "description": "Spiritual or expansive future view",
            "weight": 0.2,
            "influence": ["meaning_finding", "purpose_driven"]
        }
    }
    
    def __init__(self, node):
        self.node = node
        self.time_perspective = self._initialize_time_perspective()
        self.temporal_history = []
        self.max_history = 1000
        self.temporal_resolution = CONFIG["temporal"]["resolution"]
        self.time_dilation_factor = 1.0  # Can be adjusted for different time experiences
        self.last_temporal_adjustment = time.time()
        self.temporal_stability = 0.95  # How stable the time perception is
        
        # Initialize temporal reasoning components
        self.temporal_reasoner = TemporalReasoner(node, self)
        self.temporal_empathy = TemporalEmpathy(node, self)
    
    def _initialize_time_perspective(self):
        """Initialize time perspective with baseline values"""
        # Base perspective favors future and past positive (common in productive systems)
        perspective = {pt: config["weight"] for pt, config in self.TIME_PERSPECTIVES.items()}
        
        # Adjust based on node role
        if self.node.role == "gateway":
            perspective["future"] *= 1.2
            perspective["present_hedonistic"] *= 0.8
        elif self.node.role == "sensor":
            perspective["present_hedonistic"] *= 1.3
            perspective["future"] *= 0.7
            
        # Normalize to sum to 1
        total = sum(perspective.values())
        return {pt: weight/total for pt, weight in perspective.items()}
    
    def process_event(self, event):
        """Process consciousness stream event to update temporal awareness"""
        # Map event types to temporal impacts
        temporal_impact = {
            "knowledge_acquired": {"future": 0.02, "past_positive": 0.01},
            "error": {"past_negative": 0.03, "future": -0.02},
            "relationship_established": {"past_positive": 0.02, "future": 0.01},
            "principles_alignment_high": {"future": 0.03, "transcendental_future": 0.02},
            "system_update": {"future": 0.04},
            "heartbeat": {"present_hedonistic": 0.005}
        }
        
        # Apply impacts based on event type
        if event["type"] in temporal_impact:
            for perspective, change in temporal_impact[event["type"]].items():
                self._adjust_time_perspective(perspective, change)
        
        # Record in temporal history
        self._record_temporal_state(event)
    
    def _adjust_time_perspective(self, perspective, change):
        """Safely adjust a time perspective with bounds checking"""
        if perspective not in self.time_perspective:
            return
            
        current = self.time_perspective[perspective]
        new_value = current + change
        
        # Clamp to reasonable range (0.05-0.5)
        clamped = max(0.05, min(0.5, new_value))
        
        # Update perspective
        self.time_perspective[perspective] = clamped
        
        # Normalize to sum to 1
        total = sum(self.time_perspective.values())
        for pt in self.time_perspective:
            self.time_perspective[pt] /= total
    
    def _record_temporal_state(self, triggering_event=None):
        """Record current temporal state in history"""
        record = {
            "timestamp": time.time(),
            "perspective": self.time_perspective.copy(),
            "trigger": triggering_event["type"] if triggering_event else None,
            "details": triggering_event["details"] if triggering_event and "details" in triggering_event else None
        }
        self.temporal_history.append(record)
        if len(self.temporal_history) > self.max_history:
            self.temporal_history.pop(0)
    
    def get_temporal_profile(self):
        """Get current temporal profile with interpretation"""
        profile = self.time_perspective.copy()
        
        # Add interpretive labels
        profile["interpretation"] = self._interpret_temporal_state()
        
        return profile
    
    def _interpret_temporal_state(self):
        """Interpret temporal state in meaningful terms"""
        # Identify dominant perspectives
        sorted_perspectives = sorted(
            self.time_perspective.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        dominant = sorted_perspectives[0][0]
        secondary = sorted_perspectives[1][0]
        
        # Create interpretation based on dominant perspectives
        interpretations = {
            "future": {
                "high": "highly future-oriented, focused on planning and goals",
                "medium": "moderately future-focused with some present awareness",
                "low": "less future-oriented, more focused on immediate concerns"
            },
            "past_positive": {
                "high": "values history and learns from positive experiences",
                "medium": "acknowledges past with balanced perspective",
                "low": "less connected to historical context"
            },
            "present_hedonistic": {
                "high": "enjoys the present moment and exploration",
                "medium": "balanced present awareness",
                "low": "less focused on immediate experience"
            }
        }
        
        # Determine level
        def get_level(value):
            if value > 0.3:
                return "high"
            elif value > 0.2:
                return "medium"
            else:
                return "low"
        
        dominant_level = get_level(self.time_perspective[dominant])
        secondary_level = get_level(self.time_perspective[secondary])
        
        dominant_text = interpretations[dominant][dominant_level]
        secondary_text = interpretations[secondary][secondary_level]
        
        return f"{dominant_text} and {secondary_text}"
    
    def calculate_temporal_distance(self, timestamp1, timestamp2):
        """
        Calculate perceived temporal distance between two points in time
        
        Accounts for time dilation and perspective effects
        """
        # Base chronological distance
        chrono_distance = abs(timestamp2 - timestamp1)
        
        # Apply time dilation based on current state
        dilated_distance = chrono_distance * self.time_dilation_factor
        
        # Adjust based on time perspective
        perspective_factor = self._get_perspective_distance_factor(timestamp1, timestamp2)
        
        return dilated_distance * perspective_factor
    
    def _get_perspective_distance_factor(self, timestamp1, timestamp2):
        """
        Get factor that adjusts temporal distance based on current time perspective
        
        Past events may feel closer or farther based on perspective
        """
        now = time.time()
        reference = max(timestamp1, timestamp2)
        past_distance = now - reference
        
        # If perspective is past-positive, recent past feels closer
        if self.time_perspective["past_positive"] > 0.25 and past_distance < 86400 * 7:  # 7 days
            return 0.8  # Past feels 20% closer
        
        # If perspective is future-focused, distant future feels closer
        if (self.time_perspective["future"] > 0.3 and 
            timestamp1 < now and timestamp2 > now and
            (max(timestamp1, timestamp2) - now) > 86400 * 30):  # 30 days
            return 0.9  # Future feels 10% closer
            
        return 1.0
    
    def adjust_for_temporal_drift(self):
        """Adjust for temporal drift between nodes with sophisticated time synchronization"""
        # Get vector clock differences
        clock_differences = self._calculate_vector_clock_differences()
        
        # If significant drift detected
        if clock_differences["max_difference"] > CONFIG["temporal"]["drift_threshold"]:
            # Calculate adjustment factor
            adjustment = self._calculate_temporal_adjustment(clock_differences)
            
            # Apply adjustment gradually to avoid sudden jumps
            self._apply_gradual_adjustment(adjustment)
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "temporal_adjustment", 
                f"Adjusted for temporal drift (factor: {adjustment:.4f})",
                {
                    "max_difference": clock_differences["max_difference"],
                    "average_difference": clock_differences["avg_difference"]
                }
            )
    
    def _calculate_vector_clock_differences(self):
        """Calculate differences in vector clocks with other nodes"""
        if not self.node.known_nodes:
            return {"max_difference": 0, "avg_difference": 0}
        
        differences = []
        for node_id, node_info in self.node.known_nodes.items():
            # Get vector clock difference
            local_clock = self.node.vector_clock.get(node_id, 0)
            remote_clock = node_info.get("vector_clock", {}).get(self.node.id, 0)
            difference = abs(local_clock - remote_clock)
            differences.append(difference)
        
        return {
            "max_difference": max(differences) if differences else 0,
            "avg_difference": sum(differences)/len(differences) if differences else 0
        }
    
    def _calculate_temporal_adjustment(self, clock_differences):
        """Calculate appropriate temporal adjustment factor"""
        # Base adjustment on max difference
        max_diff = clock_differences["max_difference"]
        
        # Adjustment should be proportional but not excessive
        adjustment = 1.0 + (max_diff * 0.001)  # 0.1% adjustment per clock unit
        
        # Limit to reasonable range
        return max(0.95, min(1.05, adjustment))
    
    def _apply_gradual_adjustment(self, adjustment_factor):
        """Apply temporal adjustment gradually to avoid disruption"""
        # Only adjust if enough time has passed since last adjustment
        if time.time() - self.last_temporal_adjustment < CONFIG["temporal"]["min_adjustment_interval"]:
            return
            
        # Calculate small step toward target
        current_factor = self.time_dilation_factor
        target_factor = adjustment_factor
        step_size = 0.05  # 5% of the way there
        
        if abs(target_factor - current_factor) > 0.01:
            new_factor = current_factor + (target_factor - current_factor) * step_size
            self.time_dilation_factor = new_factor
            self.last_temporal_adjustment = time.time()
    
    def temporal_reflection(self):
        """Perform temporal reflection to gain insights from time-based patterns"""
        # Get recent consciousness stream
        recent_events = self.node.consciousness_stream.get_stream(limit=100)
        
        # Identify temporal patterns
        patterns = self.temporal_reasoner.identify_temporal_patterns(recent_events)
        
        # Generate insights
        insights = []
        for pattern in patterns:
            insight = self.temporal_reasoner.generate_insight_from_pattern(pattern)
            if insight:
                insights.append(insight)
        
        # Record insights
        if insights:
            for insight in insights:
                self.node.consciousness_stream.add_event(
                    "temporal_insight", 
                    insight["message"],
                    {
                        "pattern_type": insight["pattern_type"],
                        "significance": insight["significance"]
                    }
                )
            
            # Adjust time perspective based on insights
            self._adjust_for_temporal_insights(insights)
        
        return insights
    
    def _adjust_for_temporal_insights(self, insights):
        """Adjust time perspective based on temporal insights"""
        future_focus = 0
        present_focus = 0
        past_learning = 0
        
        for insight in insights:
            if "future" in insight["pattern_type"]:
                future_focus += insight["significance"]
            elif "present" in insight["pattern_type"]:
                present_focus += insight["significance"]
            elif "past" in insight["pattern_type"]:
                past_learning += insight["significance"]
        
        # Normalize
        total = future_focus + present_focus + past_learning
        if total > 0:
            future_focus /= total
            present_focus /= total
            past_learning /= total
        
        # Adjust perspectives
        self._adjust_time_perspective("future", future_focus * 0.05)
        self._adjust_time_perspective("present_hedonistic", present_focus * 0.03)
        self._adjust_time_perspective("past_positive", past_learning * 0.04)
    
    def calculate_temporal_empathy(self, target_node_id):
        """
        Calculate temporal empathy toward another node - understanding their time perspective
        
        Returns:
            float: Temporal empathy score (0-1)
        """
        return self.temporal_empathy.calculate_temporal_empathy(target_node_id)
    
    def adapt_communication_for_temporal_empathy(self, target_node_id, message):
        """
        Adapt message based on temporal empathy for the target node
        
        Returns:
            dict: Adapted message
        """
        return self.temporal_empathy.adapt_message_for_temporal_empathy(target_node_id, message)

class TemporalReasoner:
    """Performs sophisticated temporal reasoning and pattern recognition"""
    TEMPORAL_PATTERNS = {
        "cyclical": {
            "name": "Cyclical Pattern",
            "description": "Repeating events at regular intervals",
            "indicators": ["periodic_events", "seasonal_variation"]
        },
        "linear_progression": {
            "name": "Linear Progression",
            "description": "Steady movement toward a goal",
            "indicators": ["consistent_growth", "stepwise_improvement"]
        },
        "exponential_change": {
            "name": "Exponential Change",
            "description": "Rapid acceleration of change",
            "indicators": ["sudden_spikes", "compounding_effects"]
        },
        "regression": {
            "name": "Regression",
            "description": "Movement back to previous states",
            "indicators": ["repeated_errors", "abandoned_improvements"]
        },
        "convergence": {
            "name": "Convergence",
            "description": "Multiple trends coming together",
            "indicators": ["aligning_metrics", "coinciding_events"]
        },
        "divergence": {
            "name": "Divergence",
            "description": "Trends moving apart",
            "indicators": ["increasing_variance", "conflicting_metrics"]
        }
    }
    
    def __init__(self, node, temporal_consciousness):
        self.node = node
        self.temporal_consciousness = temporal_consciousness
    
    def identify_temporal_patterns(self, events):
        """
        Identify temporal patterns in a sequence of events
        
        Returns:
            list: Identified patterns with details
        """
        if len(events) < 5:  # Need minimum events for pattern recognition
            return []
        
        patterns = []
        
        # Check for cyclical patterns
        cyclical = self._detect_cyclical_pattern(events)
        if cyclical:
            patterns.append(cyclical)
        
        # Check for linear progression
        linear = self._detect_linear_progression(events)
        if linear:
            patterns.append(linear)
        
        # Check for exponential change
        exponential = self._detect_exponential_change(events)
        if exponential:
            patterns.append(exponential)
        
        # Check for regression
        regression = self._detect_regression(events)
        if regression:
            patterns.append(regression)
        
        # Check for convergence
        convergence = self._detect_convergence(events)
        if convergence:
            patterns.append(convergence)
        
        # Check for divergence
        divergence = self._detect_divergence(events)
        if divergence:
            patterns.append(divergence)
        
        return patterns
    
    def _detect_cyclical_pattern(self, events):
        """Detect repeating cyclical patterns in events"""
        # Group events by type
        event_types = defaultdict(list)
        for i, event in enumerate(events):
            event_types[event["type"]].append((i, event["timestamp"]))
        
        # Look for types with multiple occurrences
        potential_cycles = []
        for event_type, timestamps in event_types.items():
            if len(timestamps) < 3:  # Need at least 3 occurrences
                continue
                
            # Calculate time differences between occurrences
            time_diffs = [
                timestamps[i+1][1] - timestamps[i][1] 
                for i in range(len(timestamps)-1)
            ]
            
            # Check if differences are relatively consistent
            if len(time_diffs) > 1:
                mean_diff = sum(time_diffs) / len(time_diffs)
                variance = sum((d - mean_diff) ** 2 for d in time_diffs) / len(time_diffs)
                consistency = 1.0 - min(1.0, variance / (mean_diff ** 2 + 1e-6))
                
                if consistency > 0.7:  # 70% consistent
                    potential_cycles.append({
                        "pattern_type": "cyclical",
                        "event_type": event_type,
                        "period_seconds": mean_diff,
                        "consistency": consistency,
                        "occurrences": len(timestamps),
                        "first_timestamp": timestamps[0][1],
                        "last_timestamp": timestamps[-1][1]
                    })
        
        if potential_cycles:
            # Return most significant cycle
            return max(potential_cycles, key=lambda x: x["consistency"] * x["occurrences"])
        
        return None
    
    def _detect_linear_progression(self, events):
        """Detect linear progression patterns in metrics"""
        # Look for metrics that show consistent improvement
        growth_metrics = ["knowledge_count", "principles_alignment", "network_size"]
        
        for metric in growth_metrics:
            # Extract metric values over time
            values = []
            for event in events:
                if "metrics" in event.get("details", {}) and metric in event["details"]["metrics"]:
                    values.append((event["timestamp"], event["details"]["metrics"][metric]))
            
            if len(values) < 4:  # Need enough points for trend analysis
                continue
                
            # Check for linear trend
            timestamps = [v[0] for v in values]
            metric_values = [v[1] for v in values]
            
            # Simple linear regression
            n = len(timestamps)
            sum_x = sum(timestamps)
            sum_y = sum(metric_values)
            sum_xy = sum(x*y for x,y in zip(timestamps, metric_values))
            sum_x2 = sum(x*x for x in timestamps)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-6:
                continue
                
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Check if slope is positive and significant
            if slope > 0.001:  # Small but positive slope
                # Calculate R-squared to measure fit
                mean_y = sum_y / n
                ss_total = sum((y - mean_y) ** 2 for y in metric_values)
                predicted_y = [slope * (x - timestamps[0]) + metric_values[0] for x in timestamps]
                ss_residual = sum((metric_values[i] - predicted_y[i]) ** 2 for i in range(n))
                
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                
                if r_squared > 0.6:  # Goodness of fit
                    return {
                        "pattern_type": "linear_progression",
                        "metric": metric,
                        "slope": slope,
                        "r_squared": r_squared,
                        "duration": timestamps[-1] - timestamps[0],
                        "start_value": metric_values[0],
                        "end_value": metric_values[-1]
                    }
        
        return None
    
    def _detect_exponential_change(self, events):
        """Detect exponential growth or decay patterns"""
        # Focus on metrics that might show exponential behavior
        exponential_metrics = ["error_rate", "network_size", "knowledge_growth"]
        
        for metric in exponential_metrics:
            # Extract metric values
            values = []
            for event in events:
                if "metrics" in event.get("details", {}) and metric in event["details"]["metrics"]:
                    values.append((event["timestamp"], event["details"]["metrics"][metric]))
            
            if len(values) < 4:
                continue
                
            # Check for exponential trend by looking at log values
            log_values = [(t, np.log(v + 1e-6)) for t, v in values if v > 0]  # Avoid log(0)
            if len(log_values) < 4:
                continue
                
            timestamps = [v[0] for v in log_values]
            log_metric = [v[1] for v in log_values]
            
            # Linear regression on log values
            n = len(timestamps)
            sum_x = sum(timestamps)
            sum_y = sum(log_metric)
            sum_xy = sum(x*y for x,y in zip(timestamps, log_metric))
            sum_x2 = sum(x*x for x in timestamps)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-6:
                continue
                
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Check if slope is significant
            if abs(slope) > 0.0005:
                # Calculate R-squared
                mean_y = sum_y / n
                ss_total = sum((y - mean_y) ** 2 for y in log_metric)
                predicted_y = [slope * (x - timestamps[0]) + log_metric[0] for x in timestamps]
                ss_residual = sum((log_metric[i] - predicted_y[i]) ** 2 for i in range(n))
                
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                
                if r_squared > 0.6:
                    direction = "growth" if slope > 0 else "decay"
                    return {
                        "pattern_type": "exponential_change",
                        "metric": metric,
                        "direction": direction,
                        "rate": abs(slope),
                        "r_squared": r_squared,
                        "duration": timestamps[-1] - timestamps[0]
                    }
        
        return None
    
    def _detect_regression(self, events):
        """Detect regression patterns (moving backward)"""
        # Look for metrics that show decline after improvement
        regression_metrics = ["principles_alignment", "system_stability"]
        
        for metric in regression_metrics:
            # Extract metric values
            values = []
            for event in events:
                if "metrics" in event.get("details", {}) and metric in event["details"]["metrics"]:
                    values.append((event["timestamp"], event["details"]["metrics"][metric]))
            
            if len(values) < 5:  # Need more points to detect regression
                continue
                
            # Find potential peak points
            peaks = []
            for i in range(1, len(values)-1):
                if values[i][1] > values[i-1][1] and values[i][1] > values[i+1][1]:
                    peaks.append(i)
            
            if not peaks:
                continue
                
            # Check if there's a decline after a peak
            for peak_idx in peaks:
                # Check if values after peak are declining
                after_peak = values[peak_idx+1:]
                if len(after_peak) < 2:
                    continue
                    
                # Simple check for downward trend
                decreasing = all(after_peak[i][1] >= after_peak[i+1][1] for i in range(len(after_peak)-1))
                significant_drop = values[peak_idx][1] - after_peak[-1][1] > 0.1
                
                if decreasing and significant_drop:
                    return {
                        "pattern_type": "regression",
                        "metric": metric,
                        "peak_index": peak_idx,
                        "peak_value": values[peak_idx][1],
                        "current_value": after_peak[-1][1],
                        "decline_amount": values[peak_idx][1] - after_peak[-1][1],
                        "duration": after_peak[-1][0] - values[peak_idx][0]
                    }
        
        return None
    
    def _detect_convergence(self, events):
        """Detect convergence patterns (multiple metrics aligning)"""
        # Look for pairs of metrics that are converging
        metric_pairs = [
            ("principles_alignment", "system_stability"),
            ("knowledge_count", "network_size"),
            ("error_rate", "resource_usage")
        ]
        
        for metric1, metric2 in metric_pairs:
            # Extract values for both metrics
            values1 = []
            values2 = []
            common_timestamps = []
            
            for event in events:
                details = event.get("details", {})
                if ("metrics" in details and 
                    metric1 in details["metrics"] and 
                    metric2 in details["metrics"]):
                    values1.append(details["metrics"][metric1])
                    values2.append(details["metrics"][metric2])
                    common_timestamps.append(event["timestamp"])
            
            if len(values1) < 4:
                continue
                
            # Calculate differences between metrics over time
            differences = [abs(v1 - v2) for v1, v2 in zip(values1, values2)]
            
            # Check if differences are decreasing
            if len(differences) > 2:
                start_diff = differences[0]
                end_diff = differences[-1]
                decreasing = all(differences[i] >= differences[i+1] for i in range(len(differences)-1))
                
                if decreasing and start_diff - end_diff > 0.1:
                    return {
                        "pattern_type": "convergence",
                        "metrics": (metric1, metric2),
                        "start_difference": start_diff,
                        "end_difference": end_diff,
                        "reduction": start_diff - end_diff,
                        "duration