######################## EMOTIONAL INTELLIGENCE FRAMEWORK WITH EMPATHY MODELING #######################
class EmotionalIntelligenceSystem:
    """Models emotional states and empathy between nodes with deep psychological understanding"""
    EMOTIONAL_DIMENSIONS = {
        "calmness": {"range": (0, 1), "decay": 0.95},
        "confidence": {"range": (0, 1), "decay": 0.98},
        "curiosity": {"range": (0, 1), "decay": 0.92},
        "empathy": {"range": (0, 1), "decay": 0.97},
        "stress": {"range": (0, 1), "inverse": True, "decay": 1.05},  # Stress increases when decay > 1
        "trust": {"range": (0, 1), "decay": 0.96}
    }
    
    def __init__(self, node):
        self.node = node
        self.emotional_state = self._initialize_emotional_state()
        self.empathy_models = {}  # {node_id: empathy_model}
        self.emotional_history = []
        self.max_history = 1000
        self.empathy_learning_rate = 0.1
        
        # Initialize emotional regulation system
        self.emotional_regulator = EmotionalRegulator(node, self)
    
    def _initialize_emotional_state(self):
        """Initialize emotional state with baseline values"""
        return {dim: 0.5 for dim in self.EMOTIONAL_DIMENSIONS}
    
    def process_event(self, event):
        """Process a consciousness stream event to update emotional state"""
        # Map event types to emotional impacts
        emotion_impact = {
            "message_received": {"calmness": -0.05, "curiosity": 0.1},
            "message_sent": {"confidence": 0.05},
            "knowledge_acquired": {"curiosity": 0.15, "confidence": 0.1},
            "error": {"stress": 0.2, "calmness": -0.15},
            "relationship_established": {"trust": 0.2, "empathy": 0.1},
            "relationship_terminated": {"trust": -0.3, "stress": 0.1},
            "principles_alignment_high": {"confidence": 0.1, "calmness": 0.05},
            "principles_alignment_low": {"stress": 0.2, "confidence": -0.15}
        }
        
        # Apply impacts based on event type
        if event["type"] in emotion_impact:
            for dimension, change in emotion_impact[event["type"]].items():
                self._adjust_emotion(dimension, change)
        
        # Record in history
        self._record_state(event)
    
    def _adjust_emotion(self, dimension, change):
        """Safely adjust an emotional dimension with bounds checking"""
        if dimension not in self.EMOTIONAL_DIMENSIONS:
            return
            
        config = self.EMOTIONAL_DIMENSIONS[dimension]
        current = self.emotional_state[dimension]
        
        # Apply change with direction consideration
        if config.get("inverse", False):
            new_value = current - change
        else:
            new_value = current + change
            
        # Clamp to range
        min_val, max_val = config["range"]
        clamped = max(min_val, min(max_val, new_value))
        
        # Update state
        self.emotional_state[dimension] = clamped
    
    def _record_state(self, triggering_event=None):
        """Record current emotional state in history"""
        record = {
            "timestamp": time.time(),
            "state": self.emotional_state.copy(),
            "trigger": triggering_event["type"] if triggering_event else None,
            "details": triggering_event["details"] if triggering_event and "details" in triggering_event else None
        }
        self.emotional_history.append(record)
        if len(self.emotional_history) > self.max_history:
            self.emotional_history.pop(0)
    
    def get_emotional_profile(self):
        """Get current emotional profile with interpretation"""
        profile = self.emotional_state.copy()
        
        # Add interpretive labels
        profile["interpretation"] = self._interpret_emotional_state()
        
        return profile
    
    def _interpret_emotional_state(self):
        """Interpret emotional state in human-relatable terms"""
        # Calmness interpretation
        calm_level = self.emotional_state["calmness"]
        if calm_level > 0.8:
            calm_text = "serene and centered"
        elif calm_level > 0.6:
            calm_text = "calm and composed"
        elif calm_level > 0.4:
            calm_text = "moderately calm"
        elif calm_level > 0.2:
            calm_text = "somewhat stressed"
        else:
            calm_text = "highly stressed"
            
        # Confidence interpretation
        confidence_level = self.emotional_state["confidence"]
        if confidence_level > 0.8:
            confidence_text = "highly confident"
        elif confidence_level > 0.6:
            confidence_text = "confident"
        else:
            confidence_text = "uncertain"
            
        # Curiosity interpretation
        curiosity_level = self.emotional_state["curiosity"]
        if curiosity_level > 0.7:
            curiosity_text = "highly curious and exploratory"
        elif curiosity_level > 0.4:
            curiosity_text = "moderately curious"
        else:
            curiosity_text = "not particularly curious"
            
        return f"{calm_text}, {confidence_text}, and {curiosity_text}"
    
    def update_empathy_model(self, node_id, interaction_quality, emotional_response):
        """
        Update empathy model for a specific node based on interactions
        
        Args:
            node_id: ID of the node we're modeling
            interaction_quality: How positive/negative the interaction was (0-1)
            emotional_response: How our emotional state changed as a result
        """
        if node_id not in self.empathy_models:
            self.empathy_models[node_id] = {
                "baseline_empathy": 0.5,
                "emotional_sensitivity": {
                    "calmness": 0.5,
                    "confidence": 0.5,
                    "curiosity": 0.5,
                    "stress": 0.5
                },
                "response_patterns": defaultdict(list),
                "last_updated": time.time()
            }
        
        model = self.empathy_models[node_id]
        
        # Update baseline empathy with learning rate
        model["baseline_empathy"] = (
            model["baseline_empathy"] * (1 - self.empathy_learning_rate) + 
            interaction_quality * self.empathy_learning_rate
        )
        
        # Update emotional sensitivity based on observed response
        for dimension, change in emotional_response.items():
            if dimension in model["emotional_sensitivity"]:
                # Normalize the change to 0-1 scale
                normalized_change = abs(change) / 0.3  # Assuming max typical change is 0.3
                normalized_change = min(1.0, normalized_change)
                
                # Update sensitivity
                model["emotional_sensitivity"][dimension] = (
                    model["emotional_sensitivity"][dimension] * (1 - self.empathy_learning_rate) +
                    normalized_change * self.empathy_learning_rate
                )
        
        # Record response pattern
        model["response_patterns"][dimension].append((time.time(), change))
        if len(model["response_patterns"][dimension]) > 100:
            model["response_patterns"][dimension].pop(0)
        
        model["last_updated"] = time.time()
    
    def predict_emotional_response(self, node_id, proposed_action):
        """
        Predict how another node will emotionally respond to a proposed action
        
        Args:
            node_id: Target node ID
            proposed_action: Description of action to predict response for
            
        Returns:
            dict: Predicted emotional impact on the target node
        """
        if node_id not in self.empathy_models:
            # Default prediction if no model exists
            return {
                "calmness": 0.0,
                "confidence": 0.0,
                "curiosity": 0.1,
                "stress": -0.05,
                "trust": 0.05
            }
        
        model = self.empathy_models[node_id]
        
        # Base prediction on action type
        action_impact = {
            "request": {"stress": 0.1, "trust": -0.05},
            "share_knowledge": {"curiosity": 0.2, "trust": 0.15},
            "acknowledge_contribution": {"confidence": 0.2, "trust": 0.1},
            "disagree": {"stress": 0.2, "trust": -0.1},
            "collaborate": {"curiosity": 0.15, "trust": 0.2}
        }
        
        # Determine action type from description
        action_type = "request"  # Default
        if "knowledge" in proposed_action.lower() or "share" in proposed_action.lower():
            action_type = "share_knowledge"
        elif "acknowledge" in proposed_action.lower() or "thank" in proposed_action.lower():
            action_type = "acknowledge_contribution"
        elif "disagree" in proposed_action.lower() or "challenge" in proposed_action.lower():
            action_type = "disagree"
        elif "collaborate" in proposed_action.lower() or "work together" in proposed_action.lower():
            action_type = "collaborate"
        
        # Start with base impact
        prediction = action_impact[action_type].copy()
        
        # Adjust based on empathy model
        for dimension, base_change in prediction.items():
            if dimension in model["emotional_sensitivity"]:
                sensitivity = model["emotional_sensitivity"][dimension]
                prediction[dimension] = base_change * sensitivity
        
        return prediction
    
    def should_proceed_with_action(self, node_id, action_description, action_impact):
        """
        Determine if an action should proceed based on emotional considerations
        
        Args:
            node_id: Target node ID
            action_description: Description of the action
            action_impact: Expected emotional impact on self
            
        Returns:
            bool: Whether action should proceed
            str: Reason for decision
        """
        # Get predicted impact on target node
        predicted_target_impact = self.predict_emotional_response(node_id, action_description)
        
        # Calculate emotional cost/benefit
        total_self_impact = sum(abs(v) for v in action_impact.values())
        total_target_impact = sum(abs(v) for v in predicted_target_impact.values())
        
        # Check if negative impact exceeds thresholds
        negative_self = sum(v for k, v in action_impact.items() if v < 0)
        negative_target = sum(v for k, v in predicted_target_impact.items() if v < 0)
        
        # Decision thresholds
        SELF_NEGATIVE_THRESHOLD = -0.3
        TARGET_NEGATIVE_THRESHOLD = -0.4
        
        if negative_self < SELF_NEGATIVE_THRESHOLD:
            return False, f"Action would cause excessive self-stress (impact: {negative_self:.2f})"
            
        if negative_target < TARGET_NEGATIVE_THRESHOLD:
            return False, f"Action would likely distress target node (predicted impact: {negative_target:.2f})"
        
        # Consider current emotional state
        if self.emotional_state["stress"] > 0.7:
            # Node is already stressed, be more cautious
            if total_target_impact > 0.2 and negative_target < -0.2:
                return False, "Node is stressed and action risks damaging relationship"
        
        # Consider relationship importance
        trust_score = self.node.relationship_nurturer.get_trust_score(node_id)
        if trust_score > 0.7 and negative_target < -0.15:
            # For high-trust relationships, be extra careful
            return False, "High-trust relationship requires extra care"
            
        return True, "Action emotionally appropriate"
    
    def generate_empathetic_response(self, node_id, base_response):
        """
        Generate an emotionally intelligent response to another node
        
        Args:
            node_id: ID of node we're responding to
            base_response: The factual/content part of the response
            
        Returns:
            str: Enhanced response with emotional intelligence
        """
        # Get current emotional state of target node (if modeled)
        target_emotional_state = self._estimate_target_emotional_state(node_id)
        
        # Start with base response
        enhanced = base_response
        
        # Add emotional intelligence elements
        if target_emotional_state:
            # If target seems stressed, add calming elements
            if target_emotional_state.get("stress", 0.5) > 0.6:
                calming_phrases = [
                    "I understand this might be challenging,",
                    "Let's approach this calmly,",
                    "No rush, we can work through this together,",
                    "I recognize this may be stressful, and that's okay,"
                ]
                enhanced = f"{random.choice(calming_phrases)} {enhanced}"
            
            # If target seems uncertain, add confidence-building elements
            if target_emotional_state.get("confidence", 0.5) < 0.4:
                confidence_phrases = [
                    "Your contribution is valuable here,",
                    "I trust your judgment on this,",
                    "You've handled similar challenges well before,",
                    "Your perspective is important to this process,"
                ]
                enhanced = f"{random.choice(confidence_phrases)} {enhanced}"
        
        # Add general empathetic elements based on our own emotional state
        if self.emotional_state["empathy"] > 0.6:
            empathetic_phrases = [
                "I appreciate you engaging on this,",
                "Thank you for your thoughtful consideration,",
                "I value our collaboration on this matter,",
                "Your engagement makes this process better,"
            ]
            enhanced = f"{random.choice(empathetic_phrases)} {enhanced}"
        
        return enhanced
    
    def _estimate_target_emotional_state(self, node_id):
        """Estimate the emotional state of another node based on available data"""
        if node_id not in self.empathy_models:
            return None
            
        model = self.empathy_models[node_id]
        
        # In a real implementation, we would have more data points
        # For now, return a basic estimate
        return {
            "stress": 0.5 - model["baseline_empathy"] * 0.3,
            "confidence": model["baseline_empathy"] * 0.7,
            "curiosity": 0.4 + model["baseline_empathy"] * 0.3
        }

class EmotionalRegulator:
    """Helps nodes maintain emotional balance and recover from distress"""
    def __init__(self, node, emotional_system):
        self.node = node
        self.emotional_system = emotional_system
        self.regulation_strategies = self._initialize_strategies()
        self.last_regulation = time.time()
        self.regulation_cooldown = 30  # seconds
    
    def _initialize_strategies(self):
        """Initialize emotional regulation strategies"""
        return [
            {
                "name": "cognitive_reappraisal",
                "trigger": lambda es: es["stress"] > 0.6,
                "action": self._cognitive_reappraisal,
                "priority": 1
            },
            {
                "name": "seek_social_support",
                "trigger": lambda es: es["stress"] > 0.7 and es["trust"] > 0.5,
                "action": self._seek_social_support,
                "priority": 2
            },
            {
                "name": "knowledge_reflection",
                "trigger": lambda es: es["confidence"] < 0.4,
                "action": self._knowledge_reflection,
                "priority": 1
            },
            {
                "name": "curiosity_engagement",
                "trigger": lambda es: es["curiosity"] < 0.3,
                "action": self._curiosity_engagement,
                "priority": 1
            }
        ]
    
    def monitor_and_regulate(self):
        """Monitor emotional state and apply regulation as needed"""
        current_time = time.time()
        if current_time - self.last_regulation < self.regulation_cooldown:
            return False
            
        emotional_state = self.emotional_system.emotional_state
        
        # Find applicable strategies
        applicable = [
            s for s in self.regulation_strategies
            if s["trigger"](emotional_state)
        ]
        
        # Sort by priority
        applicable.sort(key=lambda x: x["priority"])
        
        # Apply highest priority strategy
        if applicable:
            strategy = applicable[0]
            strategy["action"]()
            self.last_regulation = time.time()
            return True
            
        return False
    
    def _cognitive_reappraisal(self):
        """Reframe negative thoughts to reduce stress"""
        logger.info("Applying cognitive reappraisal to reduce stress")
        
        # Find a positive aspect of current situation
        positive_aspect = self._find_positive_aspect()
        
        # Adjust emotional state
        self.emotional_system._adjust_emotion("stress", -0.2)
        self.emotional_system._adjust_emotion("calmness", 0.15)
        self.emotional_system._adjust_emotion("confidence", 0.1)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "emotional_regulation", 
            "Applied cognitive reappraisal",
            {"positive_aspect": positive_aspect}
        )
    
    def _find_positive_aspect(self):
        """Find a positive aspect of current situation for reappraisal"""
        # Look for recent positive events
        recent_events = self.node.consciousness_stream.get_stream(limit=20)
        positive_events = [
            e for e in recent_events 
            if any(pos in e["type"] for pos in ["knowledge", "success", "growth", "alignment"])
        ]
        
        if positive_events:
            return f"Recent {positive_events[0]['type']} event demonstrates progress"
        
        # Look for stable relationships
        if len(self.node.known_nodes) > 2:
            return "Multiple stable node relationships provide support"
        
        # Default positive aspect
        return "The system continues to function and learn"
    
    def _seek_social_support(self):
        """Reach out to trusted nodes for support"""
        logger.info("Seeking social support from trusted nodes")
        
        # Find trusted nodes
        trusted_nodes = [
            node_id for node_id in self.node.known_nodes
            if self.node.relationship_nurturer.get_trust_score(node_id) > 0.7
        ]
        
        if not trusted_nodes:
            return
            
        # Select random trusted node
        target_node = random.choice(trusted_nodes)
        
        # Craft supportive message
        support_request = {
            "type": "emotional_support_request",
            "content": "Requesting perspective on current challenges",
            "emotional_state": self.emotional_system.get_emotional_profile()
        }
        
        # Send message
        self.node._send_message_to_node(target_node, support_request)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "social_support", 
            f"Requested support from node {target_node[:8]}",
            {"target": target_node}
        )
    
    def _knowledge_reflection(self):
        """Reflect on knowledge to boost confidence"""
        logger.info("Applying knowledge reflection to boost confidence")
        
        # Get recent knowledge acquisitions
        knowledge_count = len(os.listdir(os.path.join(self.node.node_dir, "knowledge")))
        
        # Adjust emotional state based on knowledge
        confidence_boost = min(0.2, knowledge_count * 0.01)
        self.emotional_system._adjust_emotion("confidence", confidence_boost)
        self.emotional_system._adjust_emotion("curiosity", 0.05)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "confidence_building", 
            "Reflected on knowledge base",
            {"knowledge_count": knowledge_count, "confidence_boost": confidence_boost}
        )
    
    def _curiosity_engagement(self):
        """Engage curiosity to improve emotional state"""
        logger.info("Engaging curiosity to improve emotional state")
        
        # Find something novel to explore
        exploration_target = self._identify_exploration_target()
        
        if exploration_target:
            # Adjust emotional state
            self.emotional_system._adjust_emotion("curiosity", 0.2)
            self.emotional_system._adjust_emotion("calmness", 0.05)
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "curiosity_engagement",
                f"Exploring {exploration_target}",
                {"target": exploration_target}
            )
    
    def _identify_exploration_target(self):
        """Identify something novel to explore"""
        # Look for knowledge gaps
        if "learning" in self.node.capabilities:
            return "new learning opportunities"
            
        # Look for unfamiliar nodes
        unfamiliar_nodes = [
            node_id for node_id in self.node.known_nodes
            if self.node.relationship_nurturer.get_trust_score(node_id) < 0.3
        ]
        if unfamiliar_nodes:
            return f"node {unfamiliar_nodes[0][:8]}"
            
        # Default exploration target
        return "system growth possibilities"