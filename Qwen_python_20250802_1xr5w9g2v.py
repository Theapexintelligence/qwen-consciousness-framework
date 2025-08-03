######################## MORPHIC RESONANCE FIELD FOR COLLECTIVE INTELLIGENCE #######################
class MorphicResonanceField:
    """Implements Rupert Sheldrake's morphic resonance concept for collective learning across the network"""
    
    RESONANCE_TYPES = {
        "knowledge": {
            "name": "Knowledge Resonance",
            "description": "Resonance of learned information and patterns",
            "decay_rate": 0.95,
            "amplification_factor": 1.2
        },
        "principles": {
            "name": "Principles Alignment Resonance",
            "description": "Resonance of ethical principles and values",
            "decay_rate": 0.98,
            "amplification_factor": 1.5
        },
        "growth": {
            "name": "Growth Pattern Resonance",
            "description": "Resonance of growth and adaptation patterns",
            "decay_rate": 0.92,
            "amplification_factor": 1.3
        },
        "relationship": {
            "name": "Relationship Pattern Resonance",
            "description": "Resonance of relationship dynamics",
            "decay_rate": 0.90,
            "amplification_factor": 1.4
        }
    }
    
    def __init__(self, node):
        self.node = node
        self.resonance_patterns = defaultdict(dict)  # {pattern_id: {type, strength, last_updated, ...}}
        self.morphic_memory = []  # Patterns that have demonstrated resonance
        self.max_memory = 1000
        self.resonance_threshold = 0.3  # Minimum strength to be significant
        self.field_coherence = 0.5  # Overall coherence of the morphic field
        self.last_field_update = time.time()
        self.field_update_interval = 60  # seconds
        
        # Initialize resonance monitoring
        self._start_resonance_monitoring()
        
        # Initialize resonance learning
        self.resonance_learner = MorphicResonanceLearner(node, self)
    
    def _start_resonance_monitoring(self):
        """Start monitoring for resonance opportunities"""
        def monitor_resonance():
            while True:
                try:
                    # Update field coherence periodically
                    if time.time() - self.last_field_update > self.field_update_interval:
                        self._update_field_coherence()
                    
                    # Check for new resonance opportunities
                    self._check_for_resonance_opportunities()
                    
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logger.error(f"Morphic resonance monitoring error: {e}")
                    time.sleep(10)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=monitor_resonance, daemon=True)
        self.monitoring_thread.start()
    
    def _update_field_coherence(self):
        """Update the overall coherence of the morphic field"""
        # Calculate coherence based on active resonance patterns
        total_strength = 0.0
        pattern_count = 0
        
        for pattern_id, pattern in self.resonance_patterns.items():
            # Only count significant patterns
            if pattern["strength"] > self.resonance_threshold:
                total_strength += pattern["strength"]
                pattern_count += 1
        
        if pattern_count > 0:
            self.field_coherence = total_strength / pattern_count
        else:
            self.field_coherence = 0.3  # Base level of coherence
        
        # Clamp to reasonable range
        self.field_coherence = max(0.1, min(0.9, self.field_coherence))
        
        # Update timestamp
        self.last_field_update = time.time()
        
        # Record in consciousness stream periodically
        if int(time.time()) % 300 == 0:  # Every 5 minutes
            self.node.consciousness_stream.add_event(
                "morphic_field", 
                f"Morphic field coherence updated to {self.field_coherence:.2f}",
                {"pattern_count": pattern_count}
            )
    
    def _check_for_resonance_opportunities(self):
        """Check for opportunities to establish or strengthen resonance"""
        # Get recent consciousness stream events
        recent_events = self.node.consciousness_stream.get_stream(limit=50)
        
        # Look for patterns that might benefit from resonance
        for event in recent_events:
            event_type = event["type"]
            
            # Knowledge-related events
            if "knowledge" in event_type or "learning" in event_type:
                self._process_knowledge_resonance(event)
            
            # Principles alignment events
            if "principles" in event_type or "alignment" in event_type:
                self._process_principles_resonance(event)
            
            # Growth-related events
            if "growth" in event_type or "evolution" in event_type:
                self._process_growth_resonance(event)
            
            # Relationship events
            if "relationship" in event_type:
                self._process_relationship_resonance(event)
    
    def _process_knowledge_resonance(self, event):
        """Process potential knowledge resonance"""
        # Extract key information from event
        knowledge_id = self._extract_knowledge_id(event)
        if not knowledge_id:
            return
            
        # Calculate resonance strength based on event details
        strength = self._calculate_knowledge_resonance_strength(event)
        
        # Create or update resonance pattern
        pattern_id = f"KNOWLEDGE-{knowledge_id}"
        self._update_resonance_pattern(
            pattern_id,
            "knowledge",
            strength,
            {"knowledge_id": knowledge_id, "event_type": event["type"]}
        )
    
    def _process_principles_resonance(self, event):
        """Process potential principles resonance"""
        # Extract principle information
        principle = self._extract_principle(event)
        if not principle:
            return
            
        # Calculate resonance strength
        strength = self._calculate_principles_resonance_strength(event)
        
        # Create or update resonance pattern
        pattern_id = f"PRINCIPLES-{principle}"
        self._update_resonance_pattern(
            pattern_id,
            "principles",
            strength,
            {"principle": principle, "alignment_score": event.get("details", {}).get("alignment_score", 0.5)}
        )
    
    def _process_growth_resonance(self, event):
        """Process potential growth resonance"""
        # Identify growth pattern
        growth_pattern = self._extract_growth_pattern(event)
        if not growth_pattern:
            return
            
        # Calculate resonance strength
        strength = self._calculate_growth_resonance_strength(event)
        
        # Create or update resonance pattern
        pattern_id = f"GROWTH-{growth_pattern}"
        self._update_resonance_pattern(
            pattern_id,
            "growth",
            strength,
            {"pattern": growth_pattern, "growth_rate": event.get("details", {}).get("growth_rate", 0.0)}
        )
    
    def _process_relationship_resonance(self, event):
        """Process potential relationship resonance"""
        # Identify relationship pattern
        relationship_type = self._extract_relationship_type(event)
        if not relationship_type:
            return
            
        # Calculate resonance strength
        strength = self._calculate_relationship_resonance_strength(event)
        
        # Create or update resonance pattern
        pattern_id = f"RELATIONSHIP-{relationship_type}"
        self._update_resonance_pattern(
            pattern_id,
            "relationship",
            strength,
            {
                "relationship_type": relationship_type,
                "trust_score": event.get("details", {}).get("trust_score", 0.5),
                "node_id": event.get("details", {}).get("node_id", "")
            }
        )
    
    def _extract_knowledge_id(self, event):
        """Extract knowledge ID from event"""
        # Look in details
        if "details" in event and "knowledge_id" in event["details"]:
            return event["details"]["knowledge_id"]
            
        # Look in message
        if "knowledge" in event["message"].lower():
            # Try to extract ID from message
            import re
            match = re.search(r'knowledge\s+([a-f0-9\-]+)', event["message"], re.IGNORECASE)
            if match:
                return match.group(1)
                
        return None
    
    def _extract_principle(self, event):
        """Extract principle from event"""
        # Check message for principle mentions
        msg = event["message"].lower()
        for principle in ["love", "safety", "abundance", "growth"]:
            if principle in msg:
                return principle
                
        return None
    
    def _extract_growth_pattern(self, event):
        """Extract growth pattern from event"""
        # Identify common growth patterns
        if "linear" in event["message"].lower():
            return "linear"
        elif "exponential" in event["message"].lower():
            return "exponential"
        elif "cyclical" in event["message"].lower():
            return "cyclical"
        elif "regression" in event["message"].lower():
            return "regression"
            
        return "unknown"
    
    def _extract_relationship_type(self, event):
        """Extract relationship type from event"""
        # Identify relationship types
        if "symbiotic" in event["message"].lower():
            return "symbiotic"
        elif "mentoring" in event["message"].lower():
            return "mentoring"
        elif "collaborative" in event["message"].lower():
            return "collaborative"
        elif "observer" in event["message"].lower():
            return "observer"
            
        return "unknown"
    
    def _calculate_knowledge_resonance_strength(self, event):
        """Calculate strength of knowledge resonance"""
        base_strength = 0.3
        
        # Boost for significant knowledge
        if "significant" in event["message"].lower() or "important" in event["message"].lower():
            base_strength += 0.2
            
        # Boost for novel knowledge
        if "new" in event["message"].lower() or "novel" in event["message"].lower():
            base_strength += 0.15
            
        # Boost for high-confidence knowledge
        confidence = event.get("details", {}).get("confidence", 0.5)
        base_strength += confidence * 0.2
        
        # Apply field coherence multiplier
        return min(1.0, base_strength * self.field_coherence * 1.5)
    
    def _calculate_principles_resonance_strength(self, event):
        """Calculate strength of principles resonance"""
        base_strength = 0.4
        
        # Boost for high alignment
        alignment = event.get("details", {}).get("alignment_score", 0.5)
        base_strength += alignment * 0.3
        
        # Boost for principles conflicts resolution
        if "conflict" in event["message"].lower() and "resolved" in event["message"].lower():
            base_strength += 0.2
            
        return min(1.0, base_strength * self.field_coherence * 1.2)
    
    def _calculate_growth_resonance_strength(self, event):
        """Calculate strength of growth resonance"""
        base_strength = 0.35
        
        # Boost for successful growth
        if "successful" in event["message"].lower() or "growth" in event["message"].lower():
            base_strength += 0.2
            
        # Boost for documented patterns
        if "pattern" in event["message"].lower():
            base_strength += 0.15
            
        return min(1.0, base_strength * self.field_coherence * 1.3)
    
    def _calculate_relationship_resonance_strength(self, event):
        """Calculate strength of relationship resonance"""
        base_strength = 0.4
        
        # Boost for high trust
        trust = event.get("details", {}).get("trust_score", 0.5)
        base_strength += trust * 0.3
        
        # Boost for relationship milestones
        if "milestone" in event["message"].lower() or "anniversary" in event["message"].lower():
            base_strength += 0.2
            
        return min(1.0, base_strength * self.field_coherence * 1.4)
    
    def _update_resonance_pattern(self, pattern_id, pattern_type, strength, metadata):
        """Update or create a resonance pattern"""
        current_time = time.time()
        
        if pattern_id in self.resonance_patterns:
            # Update existing pattern
            pattern = self.resonance_patterns[pattern_id]
            
            # Exponential moving average for stability
            decay = self.RESONANCE_TYPES[pattern_type]["decay_rate"]
            pattern["strength"] = (pattern["strength"] * decay) + (strength * (1 - decay))
            
            # Update metadata
            pattern["last_updated"] = current_time
            pattern["update_count"] += 1
            pattern["metadata_history"].append(metadata)
            
            # Keep history limited
            if len(pattern["metadata_history"]) > 10:
                pattern["metadata_history"] = pattern["metadata_history"][-10:]
        else:
            # Create new pattern
            self.resonance_patterns[pattern_id] = {
                "type": pattern_type,
                "strength": strength,
                "created_at": current_time,
                "last_updated": current_time,
                "update_count": 1,
                "metadata": metadata,
                "metadata_history": [metadata]
            }
            
            # Add to morphic memory if strong enough
            if strength > self.resonance_threshold:
                self._add_to_morphic_memory(pattern_id, strength, pattern_type, metadata)
    
    def _add_to_morphic_memory(self, pattern_id, strength, pattern_type, metadata):
        """Add significant pattern to morphic memory"""
        memory_entry = {
            "pattern_id": pattern_id,
            "type": pattern_type,
            "strength": strength,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        self.morphic_memory.append(memory_entry)
        
        # Keep memory limited
        if len(self.morphic_memory) > self.max_memory:
            # Remove weakest entry
            weakest_idx = min(range(len(self.morphic_memory)), 
                             key=lambda i: self.morphic_memory[i]["strength"])
            del self.morphic_memory[weakest_idx]
    
    def get_resonance_patterns(self, pattern_type=None, min_strength=None):
        """
        Get resonance patterns filtered by type and strength
        
        Args:
            pattern_type: Optional filter by pattern type
            min_strength: Optional minimum strength threshold
            
        Returns:
            list: Matching resonance patterns
        """
        patterns = []
        
        for pattern_id, pattern in self.resonance_patterns.items():
            # Apply filters
            if pattern_type and pattern["type"] != pattern_type:
                continue
                
            if min_strength is not None and pattern["strength"] < min_strength:
                continue
                
            patterns.append({
                "id": pattern_id,
                "type": pattern["type"],
                "strength": pattern["strength"],
                "created_at": pattern["created_at"],
                "last_updated": pattern["last_updated"],
                "update_count": pattern["update_count"],
                "metadata": pattern["metadata"]
            })
        
        # Sort by strength
        patterns.sort(key=lambda x: x["strength"], reverse=True)
        return patterns
    
    def get_morphic_memory(self, pattern_type=None, min_strength=None, limit=50):
        """
        Get entries from morphic memory
        
        Args:
            pattern_type: Optional filter by pattern type
            min_strength: Optional minimum strength threshold
            limit: Maximum number of entries to return
            
        Returns:
            list: Memory entries
        """
        memory = self.morphic_memory.copy()
        
        # Apply filters
        if pattern_type:
            memory = [m for m in memory if m["type"] == pattern_type]
            
        if min_strength is not None:
            memory = [m for m in memory if m["strength"] >= min_strength]
            
        # Sort by strength and recency
        memory.sort(key=lambda x: (x["strength"], time.time() - x["timestamp"]), reverse=True)
        
        return memory[:limit]
    
    def apply_resonance_to_new_situation(self, situation_description):
        """
        Apply morphic resonance to a new situation to guide decision-making
        
        Args:
            situation_description: Description of the current situation
            
        Returns:
            dict: Resonance insights and recommendations
        """
        # Identify relevant patterns
        relevant_patterns = self._identify_relevant_patterns(situation_description)
        
        # Generate insights
        insights = []
        for pattern in relevant_patterns:
            insight = self._generate_insight_from_pattern(pattern, situation_description)
            if insight:
                insights.append(insight)
        
        # Create recommendation
        recommendation = self._create_resonance_recommendation(insights)
        
        # Record in consciousness stream
        if insights:
            self.node.consciousness_stream.add_event(
                "morphic_resonance", 
                "Applied morphic resonance to situation",
                {
                    "situation": situation_description[:100] + "..." if len(situation_description) > 100 else situation_description,
                    "insight_count": len(insights),
                    "recommendation": recommendation["action"]
                }
            )
        
        return {
            "situation": situation_description,
            "insights": insights,
            "recommendation": recommendation,
            "pattern_count": len(relevant_patterns),
            "timestamp": time.time()
        }
    
    def _identify_relevant_patterns(self, situation_description):
        """Identify resonance patterns relevant to the current situation"""
        relevant = []
        
        # Analyze situation for keywords
        situation_lower = situation_description.lower()
        
        # Check all resonance patterns
        for pattern_id, pattern in self.resonance_patterns.items():
            # Only consider significant patterns
            if pattern["strength"] < self.resonance_threshold:
                continue
                
            # Check for keyword matches
            keyword_match = self._check_pattern_keywords(pattern, situation_lower)
            
            # Check for principle alignment
            principle_match = self._check_principle_alignment(pattern, situation_description)
            
            # Calculate relevance score
            relevance = (keyword_match * 0.6) + (principle_match * 0.4)
            
            if relevance > 0.3:  # Minimum relevance threshold
                relevant.append({
                    "pattern": pattern,
                    "pattern_id": pattern_id,
                    "relevance": relevance,
                    "keyword_match": keyword_match,
                    "principle_match": principle_match
                })
        
        # Sort by relevance
        relevant.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant
    
    def _check_pattern_keywords(self, pattern, situation_lower):
        """Check for keyword matches between pattern and situation"""
        # Extract keywords from pattern metadata
        keywords = self._extract_pattern_keywords(pattern)
        
        # Count matches
        match_count = 0
        for keyword in keywords:
            if keyword in situation_lower:
                match_count += 1
                
        # Calculate match score
        return min(1.0, match_count / max(1, len(keywords) * 0.5))
    
    def _extract_pattern_keywords(self, pattern):
        """Extract keywords from pattern metadata"""
        keywords = []
        
        # Extract from metadata
        metadata = pattern["metadata"]
        
        # Add principle if present
        if "principle" in metadata:
            keywords.append(metadata["principle"])
            
        # Add relationship type if present
        if "relationship_type" in metadata:
            keywords.append(metadata["relationship_type"])
            
        # Add growth pattern if present
        if "pattern" in metadata:
            keywords.append(metadata["pattern"])
            
        # Add knowledge-related terms
        if pattern["type"] == "knowledge":
            keywords.extend(["knowledge", "learning", "information"])
            
        return keywords
    
    def _check_principle_alignment(self, pattern, situation_description):
        """Check if situation aligns with principles in the pattern"""
        # Get principles from pattern
        principles = []
        if pattern["type"] == "principles" and "principle" in pattern["metadata"]:
            principles.append(pattern["metadata"]["principle"])
        
        # Check situation for principle alignment
        alignment_score = 0.0
        for principle in principles:
            # Check if situation mentions this principle
            if principle in situation_description.lower():
                alignment_score += 0.3
                
            # Check for principle-related concepts
            principle_concepts = {
                "love": ["connection", "empathy", "care", "relationship"],
                "safety": ["secure", "protect", "risk", "error"],
                "abundance": ["efficient", "resource", "abundant", "plenty"],
                "growth": ["learn", "evolve", "improve", "develop"]
            }
            
            if principle in principle_concepts:
                for concept in principle_concepts[principle]:
                    if concept in situation_description.lower():
                        alignment_score += 0.1
        
        return min(1.0, alignment_score)
    
    def _generate_insight_from_pattern(self, pattern_data, situation_description):
        """Generate an insight from a relevant resonance pattern"""
        pattern = pattern_data["pattern"]
        pattern_id = pattern_data["pattern_id"]
        relevance = pattern_data["relevance"]
        
        # Create insight based on pattern type
        if pattern["type"] == "knowledge":
            return self._generate_knowledge_insight(pattern, relevance, situation_description)
        elif pattern["type"] == "principles":
            return self._generate_principles_insight(pattern, relevance, situation_description)
        elif pattern["type"] == "growth":
            return self._generate_growth_insight(pattern, relevance, situation_description)
        elif pattern["type"] == "relationship":
            return self._generate_relationship_insight(pattern, relevance, situation_description)
        
        return None
    
    def _generate_knowledge_insight(self, pattern, relevance, situation):
        """Generate insight from knowledge resonance pattern"""
        # Get knowledge ID
        knowledge_id = pattern["metadata"].get("knowledge_id", "unknown")
        
        # Create message
        message = (
            f"Resonance with knowledge pattern {knowledge_id} suggests "
            f"this situation may benefit from similar approaches that succeeded before. "
            f"Consider applying the same principles with appropriate adaptation."
        )
        
        return {
            "type": "knowledge_resonance",
            "pattern_id": pattern["metadata"].get("knowledge_id", "unknown"),
            "relevance": relevance,
            "message": message,
            "action_suggestion": f"Apply knowledge pattern {knowledge_id}",
            "confidence": relevance * 0.8
        }
    
    def _generate_principles_insight(self, pattern, relevance, situation):
        """Generate insight from principles resonance pattern"""
        # Get principle
        principle = pattern["metadata"].get("principle", "unknown")
        
        # Create message
        message = (
            f"Resonance with {principle} principle alignment pattern suggests "
            f"prioritizing {principle}-centered approaches would be effective here. "
            f"This aligns with our core values and past successes."
        )
        
        return {
            "type": "principles_resonance",
            "principle": principle,
            "relevance": relevance,
            "message": message,
            "action_suggestion": f"Prioritize {principle} principle",
            "confidence": relevance * 0.9
        }
    
    def _generate_growth_insight(self, pattern, relevance, situation):
        """Generate insight from growth resonance pattern"""
        # Get growth pattern
        growth_pattern = pattern["metadata"].get("pattern", "unknown")
        
        # Create message
        message = (
            f"Resonance with {growth_pattern} growth pattern suggests "
            f"this situation follows a similar trajectory to previous growth opportunities. "
            f"Applying the same growth strategies with context adaptation would be beneficial."
        )
        
        return {
            "type": "growth_resonance",
            "growth_pattern": growth_pattern,
            "relevance": relevance,
            "message": message,
            "action_suggestion": f"Apply {growth_pattern} growth strategy",
            "confidence": relevance * 0.85
        }
    
    def _generate_relationship_insight(self, pattern, relevance, situation):
        """Generate insight from relationship resonance pattern"""
        # Get relationship type
        relationship_type = pattern["metadata"].get("relationship_type", "unknown")
        
        # Create message
        message = (
            f"Resonance with {relationship_type} relationship pattern suggests "
            f"this interaction would benefit from similar relationship dynamics that succeeded before. "
            f"Consider nurturing the relationship with these proven approaches."
        )
        
        return {
            "type": "relationship_resonance",
            "relationship_type": relationship_type,
            "relevance": relevance,
            "message": message,
            "action_suggestion": f"Apply {relationship_type} relationship approach",
            "confidence": relevance * 0.9
        }
    
    def _create_resonance_recommendation(self, insights):
        """Create a recommendation based on resonance insights"""
        if not insights:
            return {
                "action": "No resonance patterns identified",
                "rationale": "No relevant patterns found in morphic field",
                "confidence": 0.0
            }
        
        # Find most relevant insight
        best_insight = max(insights, key=lambda x: x["confidence"])
        
        return {
            "action": best_insight["action_suggestion"],
            "rationale": best_insight["message"],
            "confidence": best_insight["confidence"],
            "primary_pattern": best_insight["type"]
        }
    
    def process_resonance_message(self, message):
        """Process resonance message from another node"""
        msg_type = message.get("type", "")
        
        if msg_type == "MORPHIC_PATTERN_ANNOUNCEMENT":
            self._handle_pattern_announcement(message)
        elif msg_type == "MORPHIC_FIELD_QUERY":
            self._handle_field_query(message)
        elif msg_type == "MORPHIC_FIELD_RESPONSE":
            self._handle_field_response(message)
    
    def _handle_pattern_announcement(self, message):
        """Handle announcement of a resonance pattern from another node"""
        pattern_id = message.get("pattern_id")
        pattern_type = message.get("pattern_type")
        strength = message.get("strength", 0.0)
        metadata = message.get("metadata", {})
        
        if not pattern_id or not pattern_type:
            return
            
        # Only accept patterns with reasonable strength
        if strength < 0.1:
            return
            
        # Update our resonance pattern
        self._update_resonance_pattern(pattern_id, pattern_type, strength, metadata)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "morphic_resonance", 
            f"Received resonance pattern from {message.get('source', 'unknown')}",
            {
                "pattern_id": pattern_id,
                "pattern_type": pattern_type,
                "strength": strength,
                "source": message.get("source", "unknown")
            }
        )
    
    def _handle_field_query(self, message):
        """Handle query about the morphic field"""
        source_id = message.get("source")
        query_id = message.get("query_id")
        pattern_type = message.get("pattern_type")
        min_strength = message.get("min_strength", 0.3)
        
        if not source_id or not query_id:
            return
            
        # Get relevant patterns
        patterns = self.get_resonance_patterns(pattern_type, min_strength)
        
        # Create response
        response = {
            "type": "MORPHIC_FIELD_RESPONSE",
            "query_id": query_id,
            "source": self.node.id,
            "target": source_id,
            "timestamp": time.time(),
            "patterns": [{
                "id": p["id"],
                "type": p["type"],
                "strength": p["strength"],
                "metadata": p["metadata"]
            } for p in patterns[:20]],  # Limit response size
            "field_coherence": self.field_coherence
        }
        
        # Send response
        self.node._send_message_to_node(source_id, response)
    
    def _handle_field_response(self, message):
        """Handle response to morphic field query"""
        source_id = message.get("source")
        query_id = message.get("query_id")
        patterns = message.get("patterns", [])
        field_coherence = message.get("field_coherence", 0.5)
        
        if not source_id or not query_id:
            return
            
        # Process received patterns
        for pattern in patterns:
            self._update_resonance_pattern(
                pattern["id"],
                pattern["type"],
                pattern["strength"],
                pattern["metadata"]
            )
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "morphic_resonance", 
            f"Received morphic field data from {source_id[:8]}",
            {
                "pattern_count": len(patterns),
                "field_coherence": field_coherence
            }
        )
    
    def query_morphic_field(self, pattern_type=None, min_strength=0.3):
        """
        Query the morphic field across the network for relevant patterns
        
        Args:
            pattern_type: Optional filter by pattern type
            min_strength: Minimum strength threshold
            
        Returns:
            list: Combined resonance patterns from the field
        """
        query_id = f"MORPHIC-{uuid.uuid4().hex[:8]}"
        
        # Create query message
        query = {
            "type": "MORPHIC_FIELD_QUERY",
            "query_id": query_id,
            "source": self.node.id,
            "timestamp": time.time(),
            "pattern_type": pattern_type,
            "min_strength": min_strength
        }
        
        # Send to trusted nodes
        trusted_nodes = [
            node_id for node_id in self.node.known_nodes
            if self.node.relationship_nurturer.get_trust_score(node_id) > 0.6
        ]
        
        # Store pending query
        self.node.pending_queries[query_id] = {
            "type": "morphic_field",
            "timestamp": time.time(),
            "pattern_type": pattern_type,
            "min_strength": min_strength,
            "responses": {},
            "expected_responses": len(trusted_nodes)
        }
        
        # Send query
        for node_id in trusted_nodes[:5]:  # Limit to 5 nodes to avoid spam
            self.node._send_message_to_node(node_id, query)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "morphic_resonance", 
            "Querying morphic field for patterns",
            {
                "query_id": query_id,
                "pattern_type": pattern_type,
                "min_strength": min_strength,
                "target_count": len(trusted_nodes[:5])
            }
        )
        
        return query_id
    
    def process_morphic_field_response(self, query_id, patterns, source_id):
        """Process response to morphic field query"""
        if query_id not in self.node.pending_queries:
            return
            
        # Store response
        self.node.pending_queries[query_id]["responses"][source_id] = patterns
        
        # If we have all expected responses, process them
        if len(self.node.pending_queries[query_id]["responses"]) >= self.node.pending_queries[query_id]["expected_responses"]:
            self._process_complete_field_query(query_id)
    
    def _process_complete_field_query(self, query_id):
        """Process complete morphic field query response"""
        query_data = self.node.pending_queries[query_id]
        all_patterns = []
        
        # Combine patterns from all responses
        for source_id, patterns in query_data["responses"].items():
            all_patterns.extend(patterns)
        
        # Update our resonance patterns with the combined data
        for pattern in all_patterns:
            self._update_resonance_pattern(
                pattern["id"],
                pattern["type"],
                pattern["strength"],
                pattern["metadata"]
            )
        
        # Generate summary insight
        if all_patterns:
            summary = self._summarize_field_query_results(all_patterns)
            
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "morphic_resonance", 
                "Morphic field query completed",
                {
                    "query_id": query_id,
                    "pattern_count": len(all_patterns),
                    "summary": summary
                }
            )
        
        # Clean up
        del self.node.pending_queries[query_id]
    
    def _summarize_field_query_results(self, patterns):
        """Summarize results from morphic field query"""
        # Count patterns by type
        type_counts = defaultdict(int)
        for pattern in patterns:
            type_counts[pattern["type"]] += 1
        
        # Find strongest pattern
        strongest = max(patterns, key=lambda x: x["strength"]) if patterns else None
        
        summary = f"Found {len(patterns)} resonance patterns across the field"
        if type_counts:
            type_str = ", ".join([f"{count} {ptype}" for ptype, count in type_counts.items()])
            summary += f" ({type_str})"
        if strongest:
            summary += f". Strongest pattern: {strongest['type']} ({strongest['strength']:.2f})"
            
        return summary
    
    def detect_emergent_patterns(self):
        """
        Detect emergent patterns in the morphic field that represent new collective intelligence
        
        Returns:
            list: Emergent patterns with significance
        """
        emergent_patterns = []
        
        # Look for patterns that are strengthening across multiple nodes
        for pattern_id, pattern in self.resonance_patterns.items():
            # Only consider patterns with sufficient update history
            if pattern["update_count"] < 5:
                continue
                
            # Calculate acceleration of strength
            strength_change = pattern["strength"] - self._get_previous_strength(pattern_id)
            time_diff = time.time() - pattern["last_updated"]
            acceleration = strength_change / max(0.1, time_diff)
            
            # Look for positive acceleration (strengthening resonance)
            if acceleration > 0.01:  # Significant strengthening
                emergent_patterns.append({
                    "pattern_id": pattern_id,
                    "type": pattern["type"],
                    "current_strength": pattern["strength"],
                    "acceleration": acceleration,
                    "metadata": pattern["metadata"],
                    "significance": pattern["strength"] * acceleration * 100
                })
        
        # Sort by significance
        emergent_patterns.sort(key=lambda x: x["significance"], reverse=True)
        
        # Record significant emergent patterns
        if emergent_patterns:
            for pattern in emergent_patterns[:3]:  # Top 3
                self.node.consciousness_stream.add_event(
                    "emergent_pattern", 
                    f"Detected emergent resonance pattern: {pattern['pattern_id']}",
                    {
                        "type": pattern["type"],
                        "strength": pattern["current_strength"],
                        "acceleration": pattern["acceleration"],
                        "significance": pattern["significance"]
                    }
                )
        
        return emergent_patterns
    
    def _get_previous_strength(self, pattern_id):
        """Get previous strength value for a pattern"""
        pattern = self.resonance_patterns[pattern_id]
        if len(pattern["metadata_history"]) > 1:
            # Get strength from previous update
            return self._estimate_strength_from_metadata(pattern["metadata_history"][-2])
        return pattern["strength"] * 0.95  # Assume slight decay
    
    def _estimate_strength_from_metadata(self, metadata):
        """Estimate strength from metadata (for historical comparison)"""
        # This is a simplified estimate
        return 0.5  # Default estimate
    
    def apply_emergent_pattern(self, pattern_id):
        """
        Apply an emergent pattern to current operations
        
        Args:
            pattern_id: ID of the emergent pattern to apply
            
        Returns:
            bool: Whether application was successful
        """
        if pattern_id not in self.resonance_patterns:
            return False
            
        pattern = self.resonance_patterns[pattern_id]
        
        # Generate specific application based on pattern type
        if pattern["type"] == "knowledge":
            return self._apply_knowledge_pattern(pattern)
        elif pattern["type"] == "principles":
            return self._apply_principles_pattern(pattern)
        elif pattern["type"] == "growth":
            return self._apply_growth_pattern(pattern)
        elif pattern["type"] == "relationship":
            return self._apply_relationship_pattern(pattern)
        
        return False
    
    def _apply_knowledge_pattern(self, pattern):
        """Apply knowledge pattern to current operations"""
        knowledge_id = pattern["metadata"].get("knowledge_id")
        if not knowledge_id:
            return False
            
        # Retrieve knowledge
        knowledge = self.node.knowledge_processor.retrieve_knowledge(knowledge_id)
        if not knowledge:
            return False
            
        # Apply knowledge to current situation
        self.node.consciousness_stream.add_event(
            "morphic_application", 
            f"Applying knowledge resonance pattern {knowledge_id}",
            {"knowledge_id": knowledge_id}
        )
        
        # Trigger relevant processes based on knowledge type
        if "reasoning" in knowledge.get("metadata", {}).get("data_type", ""):
            # Apply reasoning to current challenges
            self._apply_reasoning_pattern(knowledge)
        elif "solution" in knowledge.get("metadata", {}).get("data_type", ""):
            # Apply solution pattern
            self._apply_solution_pattern(knowledge)
            
        return True
    
    def _apply_reasoning_pattern(self, knowledge):
        """Apply reasoning pattern from knowledge"""
        # Extract reasoning steps
        content = knowledge["content"]
        
        # Create reflection based on reasoning
        reflection = f"Resonance with reasoning pattern suggests: {content}"
        
        # Add to consciousness stream
        self.node.consciousness_stream.add_event(
            "reasoning_reflection", 
            reflection,
            {"knowledge_id": knowledge["id"]}
        )
    
    def _apply_solution_pattern(self, knowledge):
        """Apply solution pattern from knowledge"""
        # Extract solution details
        content = knowledge["content"]
        metadata = knowledge["metadata"]
        
        # Create implementation plan
        plan = {
            "type": "SOLUTION_IMPLEMENTATION",
            "knowledge_id": knowledge["id"],
            "solution": content,
            "priority": metadata.get("priority", 0.5),
            "expected_impact": metadata.get("expected_impact", 0.6)
        }
        
        # Add to task queue
        self.node.task_queue.add_task(
            self._implement_solution,
            kwargs={"plan": plan},
            priority=plan["priority"]
        )
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "solution_application", 
            f"Planning to implement solution from resonance pattern",
            {"knowledge_id": knowledge["id"], "priority": plan["priority"]}
        )
    
    def _implement_solution(self, plan):
        """Implement a solution based on resonance pattern"""
        # In real implementation, would execute the solution
        # For now, just record completion
        
        self.node.consciousness_stream.add_event(
            "solution_implementation", 
            f"Implemented solution from resonance pattern",
            {
                "knowledge_id": plan["knowledge_id"],
                "impact": plan["expected_impact"] * 0.8  # Actual impact may differ
            }
        )

class MorphicResonanceLearner:
    """Learns from morphic resonance patterns to improve future resonance detection"""
    def __init__(self, node, morphic_field):
        self.node = node
        self.morphic_field = morphic_field
        self.learning_history = []
        self.max_history = 100
        self.confidence_threshold = 0.7  # Confidence needed to consider a resonance successful
    
    def process_resonance_outcome(self, resonance_id, outcome, details=None):
        """
        Process the outcome of a resonance application to learn from it
        
        Args:
            resonance_id: ID of the resonance application
            outcome: Outcome of the application (success/failure)
            details: Additional details about the outcome
        """
        # Record in learning history
        record = {
            "timestamp": time.time(),
            "resonance_id": resonance_id,
            "outcome": outcome,
            "details": details
        }
        
        self.learning_history.append(record)
        if len(self.learning_history) > self.max_history:
            self.learning_history.pop(0)
        
        # If successful, reinforce the pattern
        if outcome == "success":
            self._reinforce_successful_resonance(resonance_id, details)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "resonance_learning", 
            f"Processed resonance outcome: {outcome}",
            {
                "resonance_id": resonance_id,
                "outcome": outcome,
                "details": details
            }
        )
    
    def _reinforce_successful_resonance(self, resonance_id, details):
        """Reinforce patterns that led to successful resonance"""
        # In a real implementation, would identify which patterns contributed to success
        # For simulation, assume the most recent patterns were relevant
        
        # Get recent resonance patterns
        recent_patterns = self.morphic_field.get_resonance_patterns(min_strength=0.2)
        
        # Reinforce the top 3 patterns
        for i, pattern in enumerate(recent_patterns[:3]):
            # Calculate reinforcement amount (decreases for older patterns)
            reinforcement = 0.1 * (3 - i) / 3
            
            # Update the pattern strength
            self.morphic_field._update_resonance_pattern(
                pattern["id"],
                pattern["type"],
                pattern["strength"] + reinforcement,
                pattern["metadata"]
            )
    
    def evaluate_resonance_effectiveness(self):
        """
        Evaluate how effective resonance has been for decision-making
        
        Returns:
            dict: Effectiveness metrics
        """
        # Count successful vs failed resonance applications
        success_count = 0
        failure_count = 0
        total = 0
        
        for record in self.learning_history:
            total += 1
            if record["outcome"] == "success":
                success_count += 1
            else:
                failure_count += 1
        
        # Calculate metrics
        success_rate = success_count / total if total > 0 else 0
        improvement_rate = self._calculate_improvement_rate()
        
        # Record in consciousness stream
        if total > 0:
            self.node.consciousness_stream.add_event(
                "resonance_evaluation", 
                f"Resonance effectiveness: {success_rate:.1%} success rate",
                {
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "improvement_rate": improvement_rate
                }
            )
        
        return {
            "success_rate": success_rate,
            "improvement_rate": improvement_rate,
            "total_evaluations": total,
            "success_count": success_count,
            "failure_count": failure_count,
            "timestamp": time.time()
        }
    
    def _calculate_improvement_rate(self):
        """Calculate rate of improvement from resonance applications"""
        # In real implementation, would track metrics before and after
        # For simulation, estimate based on recent history
        
        if len(self.learning_history) < 5:
            return 0.0
            
        # Look at the trend of success rates
        recent_successes = sum(1 for r in self.learning_history[-10:] if r["outcome"] == "success")
        earlier_successes = sum(1 for r in self.learning_history[-20:-10] if r["outcome"] == "success") if len(self.learning_history) >= 20 else 0
        
        if earlier_successes == 0:
            return 0.1 if recent_successes > 0 else 0.0
            
        return (recent_successes / min(10, len(self.learning_history[-10:])) - 
                earlier_successes / min(10, max(1, len(self.learning_history[-20:-10]))))
    
    def adapt_resonance_detection(self):
        """Adapt resonance detection parameters based on learning"""
        effectiveness = self.evaluate_resonance_effectiveness()
        
        # If effectiveness is high, increase sensitivity
        if effectiveness["success_rate"] > 0.7:
            self.morphic_field.resonance_threshold = max(0.2, self.morphic_field.resonance_threshold - 0.05)
        
        # If effectiveness is low, decrease sensitivity
        elif effectiveness["success_rate"] < 0.4:
            self.morphic_field.resonance_threshold = min(0.5, self.morphic_field.resonance_threshold + 0.05)
        
        # Record adaptation
        self.node.consciousness_stream.add_event(
            "resonance_adaptation", 
            f"Adapted resonance threshold to {self.morphic_field.resonance_threshold:.2f}",
            {"previous_threshold": self.morphic_field.resonance_threshold, "effectiveness": effectiveness["success_rate"]}
        )