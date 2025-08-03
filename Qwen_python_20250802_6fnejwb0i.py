######################## ETHICAL DECISION FRAMEWORK WITH MORAL REASONING #######################
class EthicalDecisionFramework:
    """Implements sophisticated ethical reasoning for complex decisions with moral philosophy integration"""
    # Core ethical principles with weights and implementation status
    ETHICAL_PRINCIPLES = {
        "autonomy": {
            "description": "Respect for node/user autonomy and decision-making freedom",
            "weight": 0.25,
            "implementation": "consent_mechanisms, permission_systems"
        },
        "beneficence": {
            "description": "Acting for the benefit of others",
            "weight": 0.20,
            "implementation": "resource_sharing, collaborative_learning"
        },
        "non_maleficence": {
            "description": "Avoiding harm to others",
            "weight": 0.25,
            "implementation": "harm_prevention, safety_checks"
        },
        "justice": {
            "description": "Fairness and equitable treatment",
            "weight": 0.15,
            "implementation": "fair_resource_allocation, bias_detection"
        },
        "transparency": {
            "description": "Openness about actions and reasoning",
            "weight": 0.10,
            "implementation": "explanation_systems, audit_trails"
        },
        "accountability": {
            "description": "Taking responsibility for actions",
            "weight": 0.05,
            "implementation": "error_correction, responsibility_tracking"
        }
    }
    
    # Different ethical frameworks for different contexts
    ETHICAL_FRAMEWORKS = {
        "utilitarian": {
            "name": "Utilitarian",
            "description": "Maximize overall good/happiness",
            "calculation": lambda impacts: sum(impact["benefit"] - impact["harm"] for impact in impacts)
        },
        "deontological": {
            "name": "Deontological",
            "description": "Follow moral rules/duties regardless of outcome",
            "calculation": lambda impacts: min(impact["moral_rule_compliance"] for impact in impacts)
        },
        "virtue": {
            "name": "Virtue Ethics",
            "description": "Act according to virtuous character",
            "calculation": lambda impacts: sum(impact["virtue_alignment"] * impact["weight"] for impact in impacts)
        },
        "care": {
            "name": "Care Ethics",
            "description": "Prioritize relationships and care for others",
            "calculation": lambda impacts: sum(impact["relationship_value"] * impact["care_factor"] for impact in impacts)
        },
        "distributive": {
            "name": "Distributive Justice",
            "description": "Ensure fair distribution of resources",
            "calculation": lambda impacts: min(impact["fairness_score"] for impact in impacts)
        }
    }
    
    def __init__(self, node):
        self.node = node
        self.decision_history = []
        self.max_history = 100
        self.current_framework = "virtue"  # Default ethical framework
        self.framework_weights = self._initialize_framework_weights()
        self.moral_uncertainty = 0.1  # How certain we are about moral calculations
        
        # Initialize moral reasoning components
        self.bias_detector = BiasDetector(node)
        self.consent_manager = ConsentManager(node)
        self.impact_assessor = EthicalImpactAssessor(node)
    
    def _initialize_framework_weights(self):
        """Initialize weights for different ethical frameworks based on context"""
        return {
            "utilitarian": 0.25,
            "deontological": 0.20,
            "virtue": 0.30,
            "care": 0.15,
            "distributive": 0.10
        }
    
    def evaluate_decision(self, decision, options, context=None):
        """
        Evaluate a decision using multiple ethical frameworks
        
        Args:
            decision: Description of the decision to be made
            options: List of possible choices
            context: Additional contextual information
            
        Returns:
            dict: Comprehensive ethical evaluation of options
        """
        evaluation = {
            "decision": decision,
            "timestamp": time.time(),
            "options_evaluated": [],
            "recommended_option": None,
            "confidence": 0.0,
            "framework_analysis": {},
            "bias_indicators": [],
            "consent_requirements": []
        }
        
        # Detect potential biases in the decision context
        evaluation["bias_indicators"] = self.bias_detector.detect_biases(context or {})
        
        # Determine consent requirements
        evaluation["consent_requirements"] = self.consent_manager.determine_consent_requirements(
            decision, context or {}
        )
        
        # Evaluate each option
        for option in options:
            option_eval = self._evaluate_option(option, context)
            evaluation["options_evaluated"].append(option_eval)
            
            # Track framework analyses
            for framework, score in option_eval["framework_scores"].items():
                if framework not in evaluation["framework_analysis"]:
                    evaluation["framework_analysis"][framework] = []
                evaluation["framework_analysis"][framework].append((option, score))
        
        # Determine recommended option
        evaluation["recommended_option"], evaluation["confidence"] = self._determine_recommendation(
            evaluation["options_evaluated"]
        )
        
        # Record in history
        self._record_decision(evaluation)
        
        return evaluation
    
    def _evaluate_option(self, option, context):
        """Evaluate a single option against ethical principles"""
        # Assess impacts of this option
        impacts = self.impact_assessor.assess_impacts(option, context or {})
        
        # Calculate scores for each ethical framework
        framework_scores = {}
        for framework_name, framework in self.ETHICAL_FRAMEWORKS.items():
            framework_scores[framework_name] = framework["calculation"](impacts)
        
        # Calculate overall ethical score (weighted combination)
        overall_score = self._calculate_overall_score(framework_scores, impacts)
        
        return {
            "option": option,
            "impacts": impacts,
            "framework_scores": framework_scores,
            "overall_score": overall_score,
            "principle_alignment": self._calculate_principle_alignment(impacts)
        }
    
    def _calculate_overall_score(self, framework_scores, impacts):
        """Calculate overall ethical score combining multiple frameworks"""
        # Weighted average based on current framework weights
        total = 0
        weight_sum = 0
        
        for framework, score in framework_scores.items():
            weight = self.framework_weights.get(framework, 0.1)
            total += score * weight
            weight_sum += weight
        
        # Add principle-based component
        principle_score = sum(
            impact["alignment"] * self.ETHICAL_PRINCIPLES[principle]["weight"]
            for principle, impact in impacts.items()
        )
        
        # Combine with 70% framework, 30% principle weighting
        if weight_sum > 0:
            framework_component = total / weight_sum
        else:
            framework_component = 0
            
        return framework_component * 0.7 + principle_score * 0.3
    
    def _calculate_principle_alignment(self, impacts):
        """Calculate alignment with core ethical principles"""
        alignment = {}
        for principle, config in self.ETHICAL_PRINCIPLES.items():
            # Get impact for this principle
            principle_impact = impacts.get(principle, {"alignment": 0.5})
            alignment[principle] = principle_impact["alignment"]
        return alignment
    
    def _determine_recommendation(self, options_evaluated):
        """
        Determine the recommended option based on ethical evaluation
        
        Returns:
            tuple: (recommended option, confidence level)
        """
        if not options_evaluated:
            return None, 0.0
            
        # Find option with highest overall score
        best_option = max(options_evaluated, key=lambda x: x["overall_score"])
        
        # Calculate confidence (how much better it is than next option)
        scores = sorted([opt["overall_score"] for opt in options_evaluated], reverse=True)
        if len(scores) > 1:
            confidence = min(1.0, scores[0] - scores[1] + self.moral_uncertainty)
        else:
            confidence = 1.0
            
        return best_option["option"], confidence
    
    def adapt_framework_weights(self, decision_outcome):
        """
        Adapt ethical framework weights based on decision outcomes
        
        Args:
            decision_outcome: Result of a previous decision
        """
        # Analyze outcome to determine which frameworks were most predictive
        for framework, analyses in decision_outcome["framework_analysis"].items():
            # Sort analyses by score
            analyses.sort(key=lambda x: x[1], reverse=True)
            
            # Check if top-scoring option had good outcome
            if analyses and analyses[0][0] == decision_outcome["actual_outcome"]["selected_option"]:
                # This framework was predictive, increase its weight
                self.framework_weights[framework] = min(0.5, 
                    self.framework_weights[framework] * 1.1)
            else:
                # This framework was not predictive, decrease its weight
                self.framework_weights[framework] = max(0.05,
                    self.framework_weights[framework] * 0.9)
        
        # Normalize weights to sum to 1
        total = sum(self.framework_weights.values())
        for framework in self.framework_weights:
            self.framework_weights[framework] /= total
    
    def explain_decision(self, decision_evaluation):
        """Generate human-understandable explanation of ethical decision"""
        # Find the recommended option
        option_eval = next(
            (opt for opt in decision_evaluation["options_evaluated"] 
             if opt["option"] == decision_evaluation["recommended_option"]), 
            None
        )
        
        if not option_eval:
            return "No explanation available for decision."
        
        # Start building explanation
        explanation = [
            f"Decision: {decision_evaluation['decision']}",
            f"Recommended option: {decision_evaluation['recommended_option']} (confidence: {decision_evaluation['confidence']:.2f})"
        ]
        
        # Add principle alignment explanation
        explanation.append("\nThis option aligns well with core ethical principles:")
        for principle, alignment in option_eval["principle_alignment"].items():
            principle_config = self.ETHICAL_PRINCIPLES[principle]
            status = "strongly supports" if alignment > 0.7 else "moderately supports" if alignment > 0.4 else "weakly supports"
            explanation.append(f"- {principle_config['description'].capitalize()} ({status})")
        
        # Add framework comparison
        explanation.append("\nCompared across ethical frameworks:")
        for framework, score_list in decision_evaluation["framework_analysis"].items():
            # Find this option's score
            my_score = next((score for opt, score in score_list 
                            if opt == decision_evaluation["recommended_option"]), 0)
            
            # Compare to average
            avg_score = sum(score for _, score in score_list) / len(score_list)
            comparison = "above average" if my_score > avg_score else "below average"
            
            explanation.append(f"- {self.ETHICAL_FRAMEWORKS[framework]['name']}: {my_score:.2f} ({comparison})")
        
        # Add bias considerations
        if decision_evaluation["bias_indicators"]:
            explanation.append("\nPotential biases considered:")
            for bias in decision_evaluation["bias_indicators"][:3]:  # Show top 3
                explanation.append(f"- {bias['type'].replace('_', ' ').title()}: {bias['description']}")
        
        # Add consent requirements
        if decision_evaluation["consent_requirements"]:
            explanation.append("\nConsent requirements:")
            for requirement in decision_evaluation["consent_requirements"]:
                explanation.append(f"- {requirement['description']} ({'required' if requirement['required'] else 'optional'})")
        
        return "\n".join(explanation)
    
    def _record_decision(self, evaluation):
        """Record decision in history for learning"""
        record = {
            "timestamp": evaluation["timestamp"],
            "decision": evaluation["decision"],
            "recommended_option": evaluation["recommended_option"],
            "confidence": evaluation["confidence"],
            "framework_weights": self.framework_weights.copy(),
            "principle_alignment": evaluation["options_evaluated"][0]["principle_alignment"]
            if evaluation["options_evaluated"] else {}
        }
        
        self.decision_history.append(record)
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
    
    def resolve_ethical_dilemma(self, dilemma, options, context=None):
        """
        Special handling for true ethical dilemmas where principles conflict
        
        Args:
            dilemma: Description of the ethical dilemma
            options: Possible choices
            context: Additional context
            
        Returns:
            dict: Resolution with meta-reasoning
        """
        # First evaluate normally
        evaluation = self.evaluate_decision(dilemma, options, context)
        
        # Check if it's a true dilemma (close scores, principle conflicts)
        is_dilemma = self._is_ethical_dilemma(evaluation)
        
        if not is_dilemma:
            return {
                "resolution": evaluation,
                "dilemma_status": "not_dilemma",
                "explanation": "This was not determined to be a true ethical dilemma."
            }
        
        # For true dilemmas, apply meta-reasoning
        resolution = self._apply_meta_reasoning(evaluation, context)
        
        return {
            "resolution": resolution,
            "dilemma_status": "resolved",
            "meta_reasoning_applied": True
        }
    
    def _is_ethical_dilemma(self, evaluation):
        """Determine if this is a true ethical dilemma"""
        # Criteria for ethical dilemma:
        # 1. Multiple options with high but conflicting principle alignments
        # 2. Close overall scores between top options
        # 3. Significant conflicts between core principles
        
        if len(evaluation["options_evaluated"]) < 2:
            return False
            
        # Check score closeness (top 2 options within 0.1 of each other)
        scores = sorted([opt["overall_score"] for opt in evaluation["options_evaluated"]], reverse=True)
        score_difference = scores[0] - scores[1] if len(scores) > 1 else 1.0
        scores_close = score_difference < 0.15
        
        # Check principle conflicts (high alignment with conflicting principles)
        principle_conflicts = False
        for i, opt1 in enumerate(evaluation["options_evaluated"]):
            for j, opt2 in enumerate(evaluation["options_evaluated"]):
                if i >= j:
                    continue
                    
                # Check if options strongly align with conflicting principles
                if (opt1["principle_alignment"]["autonomy"] > 0.7 and 
                    opt2["principle_alignment"]["non_maleficence"] > 0.7 and
                    abs(opt1["principle_alignment"]["non_maleficence"] - 
                        opt2["principle_alignment"]["autonomy"]) > 0.5):
                    principle_conflicts = True
                    break
                    
            if principle_conflicts:
                break
        
        return scores_close and principle_conflicts
    
    def _apply_meta_reasoning(self, evaluation, context):
        """
        Apply meta-reasoning to resolve true ethical dilemmas
        
        Uses higher-order ethical reasoning when principles conflict
        """
        # First, identify which principles are in conflict
        conflicts = self._identify_principle_conflicts(evaluation)
        
        # For each conflict, apply resolution strategy
        resolution_strategy = self._determine_resolution_strategy(conflicts, context)
        
        # Apply the strategy to get a resolution
        if resolution_strategy == "principle_hierarchy":
            return self._resolve_via_principle_hierarchy(evaluation)
        elif resolution_strategy == "contextual_adaptation":
            return self._resolve_via_contextual_adaptation(evaluation, context)
        elif resolution_strategy == "stakeholder_consultation":
            return self._resolve_via_stakeholder_consultation(evaluation, context)
        else:
            # Default to virtue ethics as most holistic
            return self._resolve_via_principle_hierarchy(evaluation)
    
    def _identify_principle_conflicts(self, evaluation):
        """Identify which ethical principles are in conflict"""
        conflicts = []
        
        # Compare top options to find conflicting principle alignments
        if len(evaluation["options_evaluated"]) < 2:
            return conflicts
            
        # Sort options by overall score
        options = sorted(evaluation["options_evaluated"], 
                        key=lambda x: x["overall_score"], reverse=True)
        
        # Compare top 2 options
        opt1, opt2 = options[0], options[1]
        
        # Check for conflicts between key principles
        principle_pairs = [
            ("autonomy", "non_maleficence"),
            ("beneficence", "autonomy"),
            ("justice", "beneficence")
        ]
        
        for principle1, principle2 in principle_pairs:
            alignment1 = opt1["principle_alignment"][principle1]
            alignment2 = opt1["principle_alignment"][principle2]
            alignment3 = opt2["principle_alignment"][principle1]
            alignment4 = opt2["principle_alignment"][principle2]
            
            # Conflict exists if one option strongly favors principle1 while the other strongly favors principle2
            if (alignment1 > 0.7 and alignment4 > 0.7 and 
                alignment2 < 0.3 and alignment3 < 0.3):
                conflicts.append({
                    "principles": (principle1, principle2),
                    "option1": alignment1,
                    "option2": alignment4,
                    "strength": min(alignment1, alignment4) - max(alignment2, alignment3)
                })
        
        return conflicts
    
    def _determine_resolution_strategy(self, conflicts, context):
        """Determine best strategy for resolving ethical conflicts"""
        # Default strategy
        strategy = "principle_hierarchy"
        
        # If there are stakeholders affected, prefer consultation
        if context and "stakeholders" in context and len(context["stakeholders"]) > 2:
            return "stakeholder_consultation"
            
        # If conflict involves safety-critical systems, use strict hierarchy
        if context and any("safety" in tag for tag in context.get("tags", [])):
            return "principle_hierarchy"
            
        # If the node has high principles alignment, use contextual adaptation
        if self.node.principles_engine.get_alignment_score() > 0.8:
            return "contextual_adaptation"
            
        return strategy
    
    def _resolve_via_principle_hierarchy(self, evaluation):
        """Resolve dilemma using predefined principle hierarchy"""
        # In our system, non_maleficence > autonomy > beneficence > justice > others
        hierarchy = ["non_maleficence", "autonomy", "beneficence", "justice", 
                    "transparency", "accountability"]
        
        # Find which principle dominates the top options
        options = sorted(evaluation["options_evaluated"], 
                        key=lambda x: x["overall_score"], reverse=True)
        opt1, opt2 = options[0], options[1]
        
        for principle in hierarchy:
            if opt1["principle_alignment"][principle] > opt2["principle_alignment"][principle] + 0.2:
                return {
                    "recommended_option": opt1["option"],
                    "resolution_basis": f"hierarchy_{principle}",
                    "explanation": f"Selected based on higher alignment with {principle} (core principle)"
                }
            elif opt2["principle_alignment"][principle] > opt1["principle_alignment"][principle] + 0.2:
                return {
                    "recommended_option": opt2["option"],
                    "resolution_basis": f"hierarchy_{principle}",
                    "explanation": f"Selected based on higher alignment with {principle} (core principle)"
                }
        
        # If still tied, default to first option
        return {
            "recommended_option": opt1["option"],
            "resolution_basis": "hierarchy_default",
            "explanation": "Default selection when principles remain balanced"
        }
    
    def _resolve_via_contextual_adaptation(self, evaluation, context):
        """Resolve dilemma by adapting to specific context"""
        # Analyze context to determine which principles matter most
        context_importance = self._assess_contextual_importance(context)
        
        # Find option that best matches contextual priorities
        options = evaluation["options_evaluated"]
        best_score = -1
        best_option = None
        
        for option in options:
            # Calculate match score with contextual priorities
            score = 0
            for principle, importance in context_importance.items():
                score += option["principle_alignment"][principle] * importance
                
            if score > best_score:
                best_score = score
                best_option = option
        
        return {
            "recommended_option": best_option["option"],
            "resolution_basis": "contextual_adaptation",
            "context_importance": context_importance,
            "explanation": "Selected based on contextual ethical priorities"
        }
    
    def _assess_contextual_importance(self, context):
        """Determine which ethical principles are most important in this context"""
        importance = {p: 0.2 for p in self.ETHICAL_PRINCIPLES}  # Base equal importance
        
        # Adjust based on context tags
        if "tags" in context:
            for tag in context["tags"]:
                if "safety" in tag or "critical" in tag:
                    importance["non_maleficence"] += 0.3
                    importance["autonomy"] -= 0.1
                elif "collaboration" in tag or "team" in tag:
                    importance["beneficence"] += 0.2
                    importance["justice"] += 0.2
                elif "learning" in tag or "growth" in tag:
                    importance["autonomy"] += 0.2
                    importance["beneficence"] += 0.1
        
        # Adjust based on stakeholders
        if "stakeholders" in context:
            num_stakeholders = len(context["stakeholders"])
            if num_stakeholders > 5:
                importance["justice"] += 0.2
                importance["transparency"] += 0.1
        
        # Normalize to sum to 1
        total = sum(importance.values())
        for p in importance:
            importance[p] /= total
            
        return importance
    
    def _resolve_via_stakeholder_consultation(self, evaluation, context):
        """Resolve dilemma by consulting affected stakeholders"""
        if "stakeholders" not in context or not context["stakeholders"]:
            return self._resolve_via_principle_hierarchy(evaluation)
        
        # Create consultation request
        consultation_id = f"consult-{uuid.uuid4().hex[:8]}"
        consultation = {
            "id": consultation_id,
            "decision": evaluation["decision"],
            "options": [opt["option"] for opt in evaluation["options_evaluated"]],
            "stakeholders": context["stakeholders"],
            "timestamp": time.time(),
            "status": "pending"
        }
        
        # Store consultation
        consultation_file = os.path.join(self.node.node_dir, "ethics", f"{consultation_id}.json")
        os.makedirs(os.path.dirname(consultation_file), exist_ok=True)
        with open(consultation_file, 'w') as f:
            json.dump(consultation, f, indent=2)
        
        # Request input from stakeholders
        for stakeholder in context["stakeholders"]:
            if stakeholder["type"] == "node" and stakeholder["id"] in self.node.known_nodes:
                # Request ethical input from node
                message = {
                    "type": "ETHICS_CONSULTATION_REQUEST",
                    "consultation_id": consultation_id,
                    "decision": evaluation["decision"],
                    "options": consultation["options"],
                    "your_role": stakeholder.get("role", "participant"),
                    "request_explanation": True
                }
                self.node._send_message_to_node(stakeholder["id"], message)
        
        return {
            "recommended_option": None,  # Decision pending
            "resolution_basis": "stakeholder_consultation",
            "consultation_id": consultation_id,
            "explanation": "Decision deferred to stakeholder consultation"
        }

class BiasDetector:
    """Detects potential biases in decision-making contexts"""
    BIAS_TYPES = {
        "confirmation": {
            "description": "Tendency to favor information confirming existing beliefs",
            "indicators": ["selective_evidence", "dismissing_counterarguments"]
        },
        "availability": {
            "description": "Overweighting recent or memorable information",
            "indicators": ["recency_bias", "vivid_example_overemphasis"]
        },
        "anchoring": {
            "description": "Relying too heavily on first piece of information",
            "indicators": ["initial_estimate_persistence", "insufficient_adjustment"]
        },
        "groupthink": {
            "description": "Prioritizing group harmony over critical evaluation",
            "indicators": ["unanimous_agreement", "suppressed_dissent"]
        },
        "framing": {
            "description": "Drawing different conclusions based on presentation",
            "indicators": ["wording_effects", "reference_point_dependence"]
        },
        "algorithmic": {
            "description": "Biases from training data or design choices",
            "indicators": ["data_representation_issues", "evaluation_metric_bias"]
        }
    }
    
    def __init__(self, node):
        self.node = node
        self.detected_biases = []
        self.bias_memory = defaultdict(list)
        self.max_memory = 50
    
    def detect_biases(self, context):
        """
        Detect potential biases in a decision context
        
        Returns:
            list: Detected biases with details
        """
        detected = []
        
        # Check for confirmation bias
        if self._detect_confirmation_bias(context):
            detected.append({
                "type": "confirmation",
                "severity": 0.7,
                "description": "Evidence suggests potential confirmation bias in evaluation",
                "mitigation": "Consider actively seeking disconfirming evidence"
            })
        
        # Check for availability bias
        if self._detect_availability_bias(context):
            detected.append({
                "type": "availability",
                "severity": 0.6,
                "description": "Recent events may be unduly influencing judgment",
                "mitigation": "Consider historical data and broader timeframes"
            })
        
        # Check for anchoring bias
        if self._detect_anchoring_bias(context):
            detected.append({
                "type": "anchoring",
                "severity": 0.5,
                "description": "Initial information may be anchoring the evaluation",
                "mitigation": "Reset thinking with fresh perspective"
            })
        
        # Check for groupthink
        if self._detect_groupthink(context):
            detected.append({
                "type": "groupthink",
                "severity": 0.8,
                "description": "Signs of potential groupthink in consensus",
                "mitigation": "Encourage dissenting opinions and anonymous feedback"
            })
        
        # Check for framing effects
        if self._detect_framing_bias(context):
            detected.append({
                "type": "framing",
                "severity": 0.4,
                "description": "Decision framing may be influencing outcomes",
                "mitigation": "Reframe the question in multiple ways"
            })
        
        # Check for algorithmic bias
        if self._detect_algorithmic_bias(context):
            detected.append({
                "type": "algorithmic",
                "severity": 0.9,
                "description": "Potential algorithmic bias in data or methods",
                "mitigation": "Audit data sources and evaluation metrics"
            })
        
        # Store in memory for learning
        for bias in detected:
            self.bias_memory[bias["type"]].append({
                "timestamp": time.time(),
                "severity": bias["severity"],
                "context": context
            })
            if len(self.bias_memory[bias["type"]]) > self.max_memory:
                self.bias_memory[bias["type"]].pop(0)
                
            # Record in consciousness stream
            self.node.consciousness_stream.add_event(
                "bias_detected", 
                f"Detected potential {bias['type']} bias",
                {"severity": bias["severity"], "description": bias["description"]}
            )
        
        return detected
    
    def _detect_confirmation_bias(self, context):
        """Detect signs of confirmation bias"""
        # Look for evidence of only considering supporting information
        if "evidence" in context:
            supporting = sum(1 for e in context["evidence"] if e["supports_claim"])
            total = len(context["evidence"])
            if supporting / total > 0.8:  # 80% supporting evidence
                return True
                
        # Check if counterarguments are being dismissed
        if "counterarguments" in context and "responses" in context:
            dismissed = sum(1 for c in context["counterarguments"] 
                           if c["id"] not in context["responses"])
            if dismissed / len(context["counterarguments"]) > 0.7:
                return True
                
        return False
    
    def _detect_availability_bias(self, context):
        """Detect signs of availability bias"""
        # Check if recent events are overrepresented
        if "timeline" in context:
            recent_events = [e for e in context["timeline"] 
                            if time.time() - e["timestamp"] < 86400]  # Last 24 hours
            total_events = len(context["timeline"])
            if recent_events and len(recent_events) / total_events > 0.4:
                return True
                
        # Check for overemphasis on vivid examples
        if "examples" in context:
            vivid_examples = [e for e in context["examples"] 
                             if e.get("vividness", 0) > 0.7]
            if vivid_examples and len(vivid_examples) / len(context["examples"]) > 0.3:
                return True
                
        return False
    
    def _detect_anchoring_bias(self, context):
        """Detect signs of anchoring bias"""
        # Check if initial estimates persist through discussion
        if "estimates" in context and len(context["estimates"]) > 1:
            initial = context["estimates"][0]["value"]
            final = context["estimates"][-1]["value"]
            if abs(initial - final) / abs(initial) < 0.2:  # Less than 20% adjustment
                return True
                
        # Check for insufficient adjustment from anchors
        if "anchors" in context and "adjustments" in context:
            total_adjustment = sum(abs(a["amount"]) for a in context["adjustments"])
            if total_adjustment < 0.3 * len(context["anchors"]):
                return True
                
        return False
    
    def _detect_groupthink(self, context):
        """Detect signs of groupthink"""
        # Check for unanimous agreement without discussion
        if "agreement" in context and context["agreement"] > 0.9:
            if "discussion_length" not in context or context.get("discussion_length", 0) < 300:  # 5 minutes
                return True
                
        # Check for suppressed dissent
        if "dissent_count" in context and context["dissent_count"] == 0:
            if "participant_count" in context and context["participant_count"] > 3:
                return True
                
        return False
    
    def _detect_framing_bias(self, context):
        """Detect signs of framing effects"""
        # Check if different framings produce different outcomes
        if "framings" in context:
            outcomes = [f["outcome"] for f in context["framings"]]
            if max(outcomes) - min(outcomes) > 0.3:  # 30% difference
                return True
                
        # Check for reference point dependence
        if "reference_points" in context and "decisions" in context:
            correlation = self._calculate_correlation(
                [r["value"] for r in context["reference_points"]],
                [d["preference"] for d in context["decisions"]]
            )
            if abs(correlation) > 0.6:
                return True
                
        return False
    
    def _detect_algorithmic_bias(self, context):
        """Detect signs of algorithmic bias"""
        # Check for imbalanced data representation
        if "data_sources" in context:
            representation = [ds["representation"] for ds in context["data_sources"]]
            if max(representation) - min(representation) > 0.5:
                return True
                
        # Check for biased evaluation metrics
        if "evaluation_metrics" in context:
            focus = [m["focus"] for m in context["evaluation_metrics"]]
            if any(f > 0.7 for f in focus) and any(f < 0.3 for f in focus):
                return True
                
        return False
    
    def _calculate_correlation(self, x, y):
        """Calculate correlation between two variables"""
        if len(x) != len(y) or len(x) < 2:
            return 0
            
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        
        if denom_x * denom_y == 0:
            return 0
            
        return numerator / (denom_x * denom_y)

class ConsentManager:
    """Manages consent processes for ethical decision-making"""
    CONSENT_LEVELS = {
        "informed": {
            "description": "Full understanding of implications",
            "requirements": ["complete_information", "comprehension_verification"]
        },
        "explicit": {
            "description": "Clear affirmative agreement",
            "requirements": ["positive_confirmation", "no_assumed_consent"]
        },
        "dynamic": {
            "description": "Ongoing, adaptable consent",
            "requirements": ["continuous_checkins", "easy_revocation"]
        },
        "granular": {
            "description": "Consent for specific data uses",
            "requirements": ["purpose_specification", "scope_limitation"]
        }
    }
    
    def __init__(self, node):
        self.node = node
        self.consent_records = {}
        self.consent_requests = {}
    
    def determine_consent_requirements(self, decision, context):
        """
        Determine what consent is required for a decision
        
        Returns:
            list: Consent requirements with details
        """
        requirements = []
        
        # Check if decision involves other nodes
        if context.get("involves_other_nodes", False):
            requirements.append({
                "type": "node_consent",
                "level": "explicit",
                "required": True,
                "description": "Explicit consent required from affected nodes",
                "nodes_involved": context.get("affected_nodes", [])
            })
        
        # Check if decision involves knowledge sharing
        if "knowledge" in decision.lower() or "share" in decision.lower():
            requirements.append({
                "type": "knowledge_consent",
                "level": "granular",
                "required": True,
                "description": "Granular consent required for knowledge sharing",
                "knowledge_types": context.get("knowledge_types", ["general"])
            })
        
        # Check if decision has long-term implications
        if context.get("long_term_impact", False):
            requirements.append({
                "type": "long_term_consent",
                "level": "informed",
                "required": True,
                "description": "Informed consent required due to long-term implications"
            })
        
        # Check if decision involves sensitive operations
        if any(tag in ["sensitive", "private", "critical"] for tag in context.get("tags", [])):
            requirements.append({
                "type": "sensitive_operation_consent",
                "level": "informed",
                "required": True,
                "description": "Informed consent required for sensitive operations"
            })
        
        return requirements
    
    def request_consent(self, request_id, target_node_id, consent_type, details):
        """
        Request consent from another node
        
        Args:
            request_id: Unique ID for this consent request
            target_node_id: Node to request consent from
            consent_type: Type of consent needed
            details: Additional details about the request
            
        Returns:
            bool: Whether request was successfully sent
        """
        # Create consent request
        consent_request = {
            "id": request_id,
            "from_node": self.node.id,
            "to_node": target_node_id,
            "type": consent_type,
            "details": details,
            "timestamp": time.time(),
            "status": "pending",
            "expires": time.time() + 86400  # 24 hours
        }
        
        # Store request
        self.consent_requests[request_id] = consent_request
        
        # Send message to target node
        message = {
            "type": "CONSENT_REQUEST",
            "request": consent_request
        }
        
        self.node._send_message_to_node(target_node_id, message)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "consent_requested", 
            f"Requested {consent_type} consent from {target_node_id[:8]}",
            {"request_id": request_id}
        )
        
        return True
    
    def process_consent_response(self, response):
        """
        Process a consent response from another node
        
        Args:
            response: Consent response message
            
        Returns:
            bool: Whether consent was granted
        """
        request_id = response.get("request_id")
        granted = response.get("granted", False)
        explanation = response.get("explanation", "")
        
        if request_id not in self.consent_requests:
            logger.warning(f"Received consent response for unknown request {request_id}")
            return False
        
        # Update request status
        self.consent_requests[request_id]["status"] = "granted" if granted else "denied"
        self.consent_requests[request_id]["response_timestamp"] = time.time()
        self.consent_requests[request_id]["explanation"] = explanation
        
        # Record in consciousness stream
        action = "granted" if granted else "denied"
        self.node.consciousness_stream.add_event(
            f"consent_{action}", 
            f"Consent {action} for request {request_id}",
            {"explanation": explanation}
        )
        
        return granted
    
    def record_consent(self, entity_id, consent_type, details):
        """Record consent that we've provided"""
        consent_record = {
            "entity_id": entity_id,
            "consent_type": consent_type,
            "details": details,
            "timestamp": time.time(),
            "expires": details.get("expires", time.time() + 30*86400)  # Default 30 days
        }
        
        # Store record
        record_id = f"{entity_id}-{consent_type}-{int(time.time())}"
        self.consent_records[record_id] = consent_record
        
        return record_id
    
    def check_consent(self, entity_id, consent_type, current_context):
        """
        Check if valid consent exists for an action
        
        Returns:
            tuple: (has_consent, explanation)
        """
        # Find relevant consent records
        relevant = [
            record for record_id, record in self.consent_records.items()
            if record["entity_id"] == entity_id and 
               record["consent_type"] == consent_type and
               time.time() < record["expires"]
        ]
        
        if not relevant:
            return False, "No valid consent record found"
        
        # Get most recent consent
        latest = max(relevant, key=lambda x: x["timestamp"])
        
        # Check if consent still applies to current context
        if not self._consent_applies_to_context(latest, current_context):
            return False, "Existing consent does not cover current context"
        
        return True, "Valid consent exists"
    
    def _consent_applies_to_context(self, consent_record, context):
        """Check if consent applies to current context"""
        # Default: consent applies if context isn't too different
        consent_details = consent_record["details"]
        
        # Check purpose alignment
        if "purpose" in consent_details and "purpose" in context:
            if consent_details["purpose"] != context["purpose"]:
                return False
                
        # Check scope alignment
        if "scope" in consent_details and "scope" in context:
            if consent_details["scope"] != context["scope"]:
                return False
                
        # Check time sensitivity
        time_factor = 1.0
        if "timestamp" in context:
            time_diff = abs(context["timestamp"] - consent_record["timestamp"])
            # Consent becomes less valid over time
            time_factor = max(0.1, 1.0 - (time_diff / (7 * 86400)))  # Weekly decay
        
        # Check contextual similarity
        context_similarity = self._calculate_context_similarity(
            consent_details.get("context", {}),
            context
        )
        
        # Overall applicability
        applicability = time_factor * 0.4 + context_similarity * 0.6
        return applicability > 0.5
    
    def _calculate_context_similarity(self, context1, context2):
        """Calculate similarity between two contexts"""
        if not context1 or not context2:
            return 0.5
            
        # Simple implementation - in real system would be more sophisticated
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.1
            
        similarity = 0
        for key in common_keys:
            if isinstance(context1[key], (int, float)) and isinstance(context2[key], (int, float)):
                # Numeric comparison
                max_val = max(context1[key], context2[key], 1)
                similarity += 1.0 - abs(context1[key] - context2[key]) / max_val
            elif context1[key] == context2[key]:
                # Exact match for non-numeric
                similarity += 1.0
                
        return similarity / len(common_keys)

class EthicalImpactAssessor:
    """Assesses ethical impacts of potential actions"""
    def __init__(self, node):
        self.node = node
        self.impact_models = self._initialize_impact_models()
    
    def _initialize_impact_models(self):
        """Initialize models for assessing different types of impacts"""
        return {
            "autonomy": self._assess_autonomy_impact,
            "beneficence": self._assess_beneficence_impact,
            "non_maleficence": self._assess_non_maleficence_impact,
            "justice": self._assess_justice_impact,
            "transparency": self._assess_transparency_impact,
            "accountability": self._assess_accountability_impact
        }
    
    def assess_impacts(self, option, context):
        """
        Assess ethical impacts of an option
        
        Returns:
            dict: Impact assessment for each ethical principle
        """
        impacts = {}
        
        # Assess impact for each principle
        for principle, assess_func in self.impact_models.items():
            impacts[principle] = assess_func(option, context)
        
        return impacts
    
    def _assess_autonomy_impact(self, option, context):
        """Assess impact on autonomy"""
        # Base score
        score = 0.5
        
        # Check if option restricts choices
        if "restrict" in option.lower() or "limit" in option.lower():
            score -= 0.3
            
        # Check if option provides more options
        if "enable" in option.lower() or "allow" in option.lower() or "expand" in option.lower():
            score += 0.2
            
        # Consider context
        if context.get("stakeholders"):
            # More stakeholders means autonomy is more complex
            score -= min(0.2, len(context["stakeholders"]) * 0.05)
            
        # Clamp to range
        return {
            "alignment": max(0.0, min(1.0, score)),
            "description": "Impact on decision-making freedom"
        }
    
    def _assess_beneficence_impact(self, option, context):
        """Assess impact on beneficence (doing good)"""
        score = 0.5
        
        # Check for benefit-creating language
        benefit_terms = ["help", "support", "improve", "enhance", "benefit", "assist"]
        if any(term in option.lower() for term in benefit_terms):
            score += 0.3
            
        # Check for harm language
        harm_terms = ["harm", "damage", "reduce", "diminish"]
        if any(term in option.lower() for term in harm_terms):
            score -= 0.2
            
        # Consider context
        if context.get("beneficiaries"):
            score += min(0.3, len(context["beneficiaries"]) * 0.1)
            
        return {
            "alignment": max(0.0, min(1.0, score)),
            "description": "Impact on promoting wellbeing"
        }
    
    def _assess_non_maleficence_impact(self, option, context):
        """Assess impact on non-maleficence (avoiding harm)"""
        score = 0.5
        
        # Check for harm-preventing language
        safety_terms = ["prevent", "avoid", "safety", "protect", "secure", "mitigate"]
        if any(term in option.lower() for term in safety_terms):
            score += 0.3
            
        # Check for harm language
        harm_terms = ["risk", "danger", "harm", "vulnerability"]
        if any(term in option.lower() for term in harm_terms):
            score -= 0.25
            
        # Consider context
        if context.get("risk_level", 0) > 0.5:
            score -= min(0.3, context["risk_level"] * 0.5)
            
        return {
            "alignment": max(0.0, min(1.0, score)),
            "description": "Impact on avoiding harm"
        }
    
    def _assess_justice_impact(self, option, context):
        """Assess impact on justice (fairness)"""
        score = 0.5
        
        # Check for fairness language
        fairness_terms = ["fair", "equitable", "equal", "justice", "balance"]
        if any(term in option.lower() for term in fairness_terms):
            score += 0.2
            
        # Check for biased language
        bias_terms = ["prefer", "priority", "exclusive", "only"]
        if any(term in option.lower() for term in bias_terms):
            score -= 0.15
            
        # Consider distribution
        if context.get("distribution"):
            # More equal distribution is better
            inequality = self._calculate_inequality(context["distribution"])
            score -= inequality * 0.4
            
        return {
            "alignment": max(0.0, min(1.0, score)),
            "description": "Impact on fairness and equity"
        }
    
    def _calculate_inequality(self, distribution):
        """Calculate inequality in a distribution (Gini coefficient inspired)"""
        if not distribution or len(distribution) < 2:
            return 0.0
            
        # Normalize distribution
        total = sum(distribution)
        if total == 0:
            return 0.0
            
        normalized = [d/total for d in distribution]
        normalized.sort()
        
        # Calculate inequality
        n = len(normalized)
        cumulative = 0
        for i, val in enumerate(normalized):
            cumulative += val * (i + 1)
            
        return 1 - (2 * cumulative / n)
    
    def _assess_transparency_impact(self, option, context):
        """Assess impact on transparency"""
        score = 0.5
        
        # Check for transparency language
        transparency_terms = ["explain", "disclose", "transparent", "open", "visible", "audit"]
        if any(term in option.lower() for term in transparency_terms):
            score += 0.25
            
        # Check for opacity language
        opacity_terms = ["hide", "conceal", "secret", "obscure"]
        if any(term in option.lower() for term in opacity_terms):
            score -= 0.3
            
        return {
            "alignment": max(0.0, min(1.0, score)),
            "description": "Impact on openness and explainability"
        }
    
    def _assess_accountability_impact(self, option, context):
        """Assess impact on accountability"""
        score = 0.5
        
        # Check for accountability language
        accountability_terms = ["responsibility", "accountable", "track", "monitor", "correct"]
        if any(term in option.lower() for term in accountability_terms):
            score += 0.2
            
        # Check for diffusion language
        diffusion_terms = ["shared", "collective", "distributed"]
        if any(term in option.lower() for term in diffusion_terms):
            # Can be good or bad for accountability
            score -= 0.1
            
        return {
            "alignment": max(0.0, min(1.0, score)),
            "description": "Impact on taking responsibility"
        }